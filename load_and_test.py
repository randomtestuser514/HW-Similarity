import os
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Import the model and helper functions
from model import HandwritingSimilarityModel, crop_text_region
from main import load_model  # load_model is defined in main.py

# ----- Configuration -----
data_dir = "<REPLACE>"  # Directory containing image files
MODEL_SAVE_PATH = os.path.join("./saved_models", "handwriting_model_final.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SIZE = 224  # Desired size for model input

# ----- Define Evaluation Transform (without Resize) -----
# We remove Resize() here since our custom preprocessing will handle it.
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----- Custom Preprocessing Function -----
def preprocess_image(file_path, target_size=TARGET_SIZE):
    """
    Load an image, apply eval_transform, crop its text region,
    and then resize the result to (target_size, target_size).
    
    Returns:
        A torch tensor of shape (3, target_size, target_size) with values in [0,1],
        or None if processing fails.
    """
    try:
        img = Image.open(file_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

    # Apply evaluation transform (without resize)
    img_tensor = eval_transform(img)  # Shape: (3, H, W)
    # Crop the image to its text region.
    cropped = crop_text_region(img_tensor)
    # Resize the cropped image to the target size.
    processed = F.interpolate(cropped.unsqueeze(0), size=(target_size, target_size),
                              mode='bilinear', align_corners=False).squeeze(0)
    return processed

# ----- Load and Sample Data -----
# Get a list of image file paths
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

if len(image_files) == 0:
    raise ValueError("No image files found in the specified data directory.")

# Randomly sample a handful (e.g., 20 images)
sample_size = min(25, len(image_files))
sampled_files = random.sample(image_files, sample_size)
print("Selected", sample_size, "number of documents")
sorted = sampled_files.copy()
sorted.sort()
for file in sorted:
    print("\t", file)

# Randomly select a reference document from the sampled files
ref_file = random.choice(sampled_files)
print("Reference document:", ref_file)

# ----- Load Model -----
model = HandwritingSimilarityModel(latent_dim=128, hidden_dim=256, patch_size=224, num_bootstrap=20)
model = load_model(model, MODEL_SAVE_PATH, device)

# ----- Preprocess Images -----
# Create a dictionary mapping file paths to preprocessed tensors.
preprocessed = {}
for f in sampled_files:
    tensor = preprocess_image(f)
    if tensor is not None:
        preprocessed[f] = tensor
    else:
        print(f"Skipping {f} (unable to process).")

if ref_file not in preprocessed:
    ref_file = random.choice(list(preprocessed.keys()))
    print("Updated Reference document:", ref_file)
    
ref_tensor = preprocessed[ref_file].to(device)

# ----- New Inference Function (for Preprocessed Images) -----
def infer_similarity_tensor_from_preprocessed(model, tensor1, tensor2):
    """
    Compute the latent embeddings directly from preprocessed image tensors.
    
    Assumes tensor1 and tensor2 are of shape (3, 224, 224).
    This bypasses the crop/bootstrapping in the model's forward method.
    
    Returns the Euclidean distance between the embeddings.
    """
    model.eval()
    with torch.no_grad():
        # Compute embeddings using the ViT and the encoder.
        # tensor.unsqueeze(0) adds a batch dimension.
        latent1 = model.encoder(model.vit(tensor1.unsqueeze(0)))
        latent2 = model.encoder(model.vit(tensor2.unsqueeze(0)))
        distance = torch.norm(latent1 - latent2, p=2).item()
    return distance

# ----- Compare Reference Document to All Others -----
results = []  # List to hold (file, distance) tuples

for f, tensor in preprocessed.items():
    if f == ref_file:
        continue
    distance = infer_similarity_tensor_from_preprocessed(model, ref_tensor, tensor.to(device))
    results.append((f, distance))

if not results:
    print("No valid comparisons could be made.")
else:
    # Sort results by ascending distance (lower = more similar)
    results.sort(key=lambda x: x[1])
    top_k = len(results)
    print("\nTop {} most similar documents to the reference:".format(top_k))
    for i, (f, d) in enumerate(results[:top_k], 1):
        print(f"{i}. {f} (Distance: {d:.4f})")
