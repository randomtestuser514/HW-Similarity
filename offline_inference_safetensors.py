# https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k/blob/main/model.safetensors

import os
import random
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file

from model import HandwritingSimilarityModel, crop_text_region

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(
    description="Offline inference using local .safetensors for ViT and HandwritingSimilarityModel"
)
parser.add_argument("--data_dir", type=str, required=True,
                    help="Directory containing handwriting image files")
parser.add_argument("--vit_safetensors", type=str, required=True,
                    help="Path to pretrained ViT-small safetensors file")
parser.add_argument("--hw_safetensors", type=str, required=True,
                    help="Path to HandwritingSimilarityModel safetensors file")
parser.add_argument("--num_samples", type=int, default=20,
                    help="Number of random images to sample for comparison")
args = parser.parse_args()

# ----- Configuration -----
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TARGET_SIZE = 224
BATCH_SIZE = 1  # we're doing pairwise on single images

# ----- Transforms -----
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----- Preprocessing -----
def preprocess_image(path, target_size=TARGET_SIZE):
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None
    tensor = eval_transform(img)
    cropped = crop_text_region(tensor)
    return F.interpolate(cropped.unsqueeze(0), size=(target_size, target_size),
                         mode='bilinear', align_corners=False).squeeze(0)

# ----- Model Initialization and Weight Loading -----
print("[Model] Initializing architecture...")
model = HandwritingSimilarityModel(latent_dim=128, hidden_dim=256,
                                   patch_size=TARGET_SIZE, num_bootstrap=20)
# Load ViT backbone weights
print(f"[Model] Loading ViT weights from {args.vit_safetensors}...")
vit_sd = load_file(args.vit_safetensors, device=DEVICE)
model.vit.load_state_dict(vit_sd, strict=False)
# Load handwriting model weights
print(f"[Model] Loading handwriting model weights from {args.hw_safetensors}...")
hw_sd = load_file(args.hw_safetensors, device=DEVICE)
model.load_state_dict(hw_sd, strict=False)
model.to(DEVICE).eval()
print("[Model] Weights loaded. Model ready on device: %s" % DEVICE)

# ----- Load and Sample Images -----
all_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
if not all_files:
    raise ValueError(f"No image files found in {args.data_dir}")
sample_size = min(args.num_samples, len(all_files))
sampled = random.sample(all_files, sample_size)
print(f"[Data] Sampled {sample_size} images from {args.data_dir}")
for p in sampled:
    print('  ', p)
# Choose reference
ref = random.choice(sampled)
print(f"[Data] Reference image: {ref}")

# Preprocess all samples
tensors = {}
for p in sampled:
    t = preprocess_image(p)
    if t is not None:
        tensors[p] = t.to(DEVICE)
    else:
        print(f"[Data] Skipping {p}")
# Ensure reference present
if ref not in tensors:
    ref = next(iter(tensors))
    print(f"[Data] Updated reference to {ref}")
ref_tensor = tensors[ref]

# ----- Inference Function -----
def compute_distance(t1, t2):
    # Direct pass through backbone + encoder
    with torch.no_grad():
        f1 = model.vit(t1.unsqueeze(0))
        z1 = model.encoder(f1)
        f2 = model.vit(t2.unsqueeze(0))
        z2 = model.encoder(f2)
        return torch.norm(z1 - z2, p=2).item()

# ----- Compare and Rank -----
results = []
for path, tensor in tensors.items():
    if path == ref:
        continue
    dist = compute_distance(ref_tensor, tensor)
    results.append((path, dist))
# Sort ascending
results.sort(key=lambda x: x[1])

print(f"Top {len(results)} similarities to reference:")
for rank, (path, dist) in enumerate(results, start=1):
    print(f"{rank:2d}. {path} -> distance {dist:.4f}")
