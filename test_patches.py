import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

from model import bootstrap_patches, crop_text_region

# Define a simple transform that converts an image to tensor (values in [0,1]).
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def visualize_patches(patches):
    """
    Visualize a set of patches in a grid with a black background.
    
    :param patches: Tensor of shape (num_patches, C, H, W)
    """
    patches_np = patches.cpu().numpy()  # Convert to numpy (num_patches, C, H, W)
    num_patches = patches_np.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_patches)))  # Square grid layout

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2), facecolor='black')
    axes = np.atleast_1d(axes).flatten()  # Ensures axes is always an array

    for i, ax in enumerate(axes):
        if i < num_patches:
            patch = patches_np[i]
            if patch.shape[0] == 1:  # Grayscale image
                ax.imshow(patch[0], cmap='gray', vmin=0, vmax=1)
            else:  # RGB image
                ax.imshow(patch.transpose(1, 2, 0))

        ax.set_xticks([])  # Remove ticks
        ax.set_yticks([])
        ax.set_frame_on(False)  # Remove borders

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between patches
    plt.suptitle("Bootstrapped Patches", color='white', fontsize=16)
    plt.show()

def process_image(image_path, patch_size=224, num_bootstrap=20):
    # Load image and convert to tensor
    image = Image.open(image_path).convert("RGB")
    tensor_image = transform(image)  # shape: (C, H, W) with values in [0,1]
    
    print(image.size)
    # Optionally crop the text region first.
    cropped = crop_text_region(tensor_image)
    # Extract patches using the bootstrap method.
    patches = bootstrap_patches(cropped, patch_size=patch_size, num_bootstrap=num_bootstrap)
    return cropped, patches

if __name__ == '__main__':
    # Update the folder path to your image folder.
    image_folder = "<REPLACE>"
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    # Choose a few images to test patch generation.
    for image_path in image_files[:20]:
        print(f"Processing image: {image_path}")
        cropped, patches = process_image(image_path, patch_size=224, num_bootstrap=20)
        print(f"Generated {patches.shape[0]} patches.")
        visualize_patches(patches)
