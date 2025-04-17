import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from data import prepare_data
from model import HandwritingSimilarityModel

# ----- Dataset Class -----
class HandwritingPairDataset(Dataset):
    """
    Loads pairs of handwriting images along with a label (1 if same writer, 0 otherwise).
    """
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path1, path2, label = self.data_list[idx]
        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        label = torch.tensor(label, dtype=torch.float32)
        return image1, image2, label

# ----- Transforms -----
# For handwriting, converting to grayscale can help; here we convert to grayscale and then replicate to 3 channels.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----- Hyperparameters -----
data_directory = "./data"  # Folder containing image files.
train_ratio = 0.8
negatives_per_positive = 2
batch_size = 1  # Process one image pair at a time.
num_epochs = 20
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_DIR = "./saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MARGIN = 1.75  # Margin for contrastive loss.
EARLY_STOP_PATIENCE = 3  # Early stopping patience.

# ----- Prepare Data -----
print("[Main] Preparing data...")
train_pairs, test_pairs = prepare_data(data_directory, train_ratio, negatives_per_positive)
print(f"[Main] Training pairs: {len(train_pairs)} | Testing pairs: {len(test_pairs)}")

train_dataset = HandwritingPairDataset(train_pairs, transform=train_transform)
test_dataset = HandwritingPairDataset(test_pairs, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----- Initialize Model -----
print("[Main] Initializing model...")
model = HandwritingSimilarityModel(latent_dim=128, hidden_dim=256, patch_size=224, num_bootstrap=20)
model.to(device)

# ----- Contrastive Loss -----
def contrastive_loss(latent1, latent2, label, margin=MARGIN):
    distance = torch.norm(latent1 - latent2, p=2)
    loss_pos = label * 0.5 * distance**2
    loss_neg = (1 - label) * 0.5 * torch.clamp(margin - distance, min=0)**2
    return loss_pos + loss_neg

# ----- Training Loop with Early Stopping and Per-Epoch Saving -----
def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_eval_loss = float("inf")
    epochs_without_improve = 0

    overall_start = time.time()
    print("[Train] Starting training...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            image1, image2, label = batch
            image1 = image1.squeeze(0).to(device)
            image2 = image2.squeeze(0).to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            latent1, latent2 = model(image1, image2)
            loss = contrastive_loss(latent1, latent2, label, margin=MARGIN)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) <= 10:
                print(f"[Train] Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")

            if (batch_idx + 1) % 100 == 0:
                print(f"[Train] Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_end = time.time()
        print(f"[Train] Epoch {epoch+1} complete. Avg Train Loss: {avg_train_loss:.4f}. Time: {epoch_end - epoch_start:.2f}s")
        
        # Evaluate loss.
        eval_loss = evaluate_loss(model, test_loader, device)
        print(f"[Eval] Epoch {epoch+1}: Avg Eval Loss: {eval_loss:.4f}")

        # Save model for this epoch.
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"handwriting_model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"[Main] Saved model: {model_save_path}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOP_PATIENCE:
                print(f"[Train] Early stopping triggered after epoch {epoch+1}.")
                break

    overall_end = time.time()
    print(f"[Train] Training complete. Total time: {overall_end - overall_start:.2f}s")

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            image1, image2, label = batch
            image1 = image1.squeeze(0).to(device)
            image2 = image2.squeeze(0).to(device)
            label = label.to(device)
            latent1, latent2 = model(image1, image2)
            loss = contrastive_loss(latent1, latent2, label, margin=MARGIN)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    total = 0
    correct = 0
    print("[Eval] Starting accuracy evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image1, image2, label = batch
            image1 = image1.squeeze(0).to(device)
            image2 = image2.squeeze(0).to(device)
            label = label.to(device)
            latent1, latent2 = model(image1, image2)
            distance = torch.norm(latent1 - latent2, p=2)
            predicted = 1.0 if distance < threshold else 0.0
            total += 1
            correct += (predicted == label.item())
            if (batch_idx + 1) % 10 == 0:
                print(f"[Eval] Batch {batch_idx+1}: True Label = {label.item()}, Distance = {distance.item():.4f}, Predicted = {predicted}")
    print(f"[Eval] Accuracy: {100 * correct/total:.2f}%")

def infer_similarity(model, image_path1, image_path2, transform, device, return_explain=False):
    print("[Infer] Running inference...")
    model.eval()
    with torch.no_grad():
        image1 = transform(Image.open(image_path1).convert("RGB")).to(device)
        image2 = transform(Image.open(image_path2).convert("RGB")).to(device)
        if return_explain:
            latent1, latent2, attn_weights, patches = model(image1, image2, return_attn=True)
            distance = torch.norm(latent1 - latent2, p=2)
            print(f"[Infer] Euclidean distance: {distance.item():.4f}")
            return distance.item(), latent1, latent2, attn_weights, patches
        else:
            latent1, latent2 = model(image1, image2)
            distance = torch.norm(latent1 - latent2, p=2)
            print(f"[Infer] Euclidean distance: {distance.item():.4f}")
            return distance.item(), latent1, latent2

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"[Main] Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Main] Model loaded from {path}")
    return model

# ----- Main Execution -----
if __name__ == '__main__':
    print("[Main] Starting main execution...")
    train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)
    evaluate_model(model, test_loader, device, threshold=0.5)
    final_model_path = os.path.join(MODEL_SAVE_DIR, "handwriting_model_final.pth")
    save_model(model, final_model_path)
    loaded_model = HandwritingSimilarityModel(latent_dim=128, hidden_dim=256, patch_size=224, num_bootstrap=20)
    loaded_model = load_model(loaded_model, final_model_path, device)
    # Example inference (update the paths accordingly)
    # infer_similarity(loaded_model, "path/to/test_image1.jpg", "path/to/test_image2.jpg", eval_transform, device, return_explain=True)