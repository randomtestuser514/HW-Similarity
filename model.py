import torch
import torch.nn as nn
import timm
import random
import cv2
import numpy as np

def crop_text_region(image, padding=10):
    """
    Crop the image to its text region by removing blank white space.
    
    Assumes the input is a torch tensor with values in [0,1].
    
    If the input has an extra batch dimension (i.e. shape (1, C, H, W)), it is squeezed.
    Also, if the image has 1 channel (grayscale), it is replicated to 3 channels.
    
    Returns a cropped tensor of shape (3, H_crop, W_crop) with values in [0,1].
    """
    # If there's a batch dimension, remove it.
    if image.ndim == 4:
        image = image.squeeze(0)  # now shape becomes (C, H, W)
    
    # Convert tensor to numpy array.
    image_np = image.cpu().numpy()  # expected shape: (C, H, W) or (H, W)
    
    # If the image is 2D (H, W), add a channel dimension.
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=0)  # now (1, H, W)
    
    # If image has 1 channel, replicate to get 3 channels.
    if image_np.shape[0] == 1:
        image_np = np.repeat(image_np, 3, axis=0)
    elif image_np.shape[0] != 3:
        raise ValueError("Input image must have either 1 or 3 channels.")
    
    # Transpose to (H, W, C)
    image_np = np.transpose(image_np, (1, 2, 0))
    
    # Scale to 0-255 and convert to uint8.
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert to grayscale for detection.
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Invert and threshold using Otsu's method.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours corresponding to text regions.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # No text detected; return original.
    
    # Compute the bounding box covering all contours.
    x_min = min(cv2.boundingRect(cnt)[0] for cnt in contours)
    y_min = min(cv2.boundingRect(cnt)[1] for cnt in contours)
    x_max = max(cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt in contours)
    y_max = max(cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt in contours)
    
    # Add padding.
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_np.shape[1], x_max + padding)
    y_max = min(image_np.shape[0], y_max + padding)
    
    cropped_np = image_np[y_min:y_max, x_min:x_max]
    
    # Convert back to tensor.
    cropped = torch.from_numpy(cropped_np).float() / 255.0
    # Ensure output shape is (3, H, W).
    if cropped.ndim == 2:
        cropped = cropped.unsqueeze(0).repeat(3, 1, 1)
    elif cropped.shape[0] != 3:
        # If channels are last dimension, permute them.
        if cropped.ndim == 3 and cropped.shape[2] == 3:
            cropped = cropped.permute(2, 0, 1)
        else:
            raise ValueError("Unexpected shape for cropped image.")
    return cropped

def bootstrap_patches(image, patch_size=224, num_bootstrap=20, 
                      brightness_threshold=0.95, std_threshold=0.075,
                      max_attempts=15):
    """
    Crop the input image to its text region and then randomly sample patches 
    (with replacement) that contain information (i.e. are not almost blank).

    Args:
      image: torch tensor of shape (C, H, W) with values in [0,1].
      patch_size: desired patch size.
      num_bootstrap: number of patches to sample.
      brightness_threshold: maximum allowed mean brightness for a patch to be considered blank.
      std_threshold: minimum allowed standard deviation for a patch to be considered informative.
      max_attempts: maximum sampling attempts per patch.
    
    Returns a tensor of shape (num_patches, C, patch_size, patch_size).
    """
    cropped_image = crop_text_region(image)
    C, H, W = cropped_image.shape
    patches = []
    attempts = 0
    while len(patches) < num_bootstrap and attempts < num_bootstrap * max_attempts:
        if H < patch_size or W < patch_size:
            patches.append(cropped_image)
            break
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)
        patch = cropped_image[:, top:top+patch_size, left:left+patch_size]
        # Check if the patch is informative based on mean and std.
        patch_mean = patch.mean().item()
        patch_std = patch.std().item()
        if patch_mean < brightness_threshold or patch_std > std_threshold:
            patches.append(patch)
        attempts += 1
    if patches:
        return torch.stack(patches)
    else:
        new_brightness_threshold = brightness_threshold + 0.05
        new_std_threshold = std_threshold - 0.02
        return bootstrap_patches(image, patch_size, num_bootstrap, new_brightness_threshold, new_std_threshold, max_attempts)

class HandwritingSimilarityModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, patch_size=224, num_bootstrap=20):
        super(HandwritingSimilarityModel, self).__init__()
        self.patch_size = patch_size
        self.num_bootstrap = num_bootstrap
        # Load a pre-trained ViT model (fine-tuning allowed).
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True)
        self.vit.reset_classifier(0)  # Remove classification head.
        feature_dim = self.vit.embed_dim  # e.g., 384
        
        # Encoder: reduce the ViT features to a compact latent space.
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Attention pooling: compute a weight for each patch.
        self.attention = nn.Linear(latent_dim, 1)
    
    def forward_image(self, image, return_attn=False):
        """
        Process a single handwriting image:
         - Crop to text region.
         - Bootstrap sample patches.
         - Process patches via the pre-trained ViT and encoder.
         - Aggregate using attention pooling.
        If return_attn is True, also return the attention weights and patches.
        """
        patches = bootstrap_patches(image, patch_size=self.patch_size, num_bootstrap=self.num_bootstrap)
        features = self.vit(patches)            # (num_patches, feature_dim)
        latent_patches = self.encoder(features)   # (num_patches, latent_dim)
        attn_weights = torch.softmax(self.attention(latent_patches), dim=0)  # (num_patches, 1)
        global_latent = torch.sum(attn_weights * latent_patches, dim=0)  # (latent_dim,)
        if return_attn:
            return global_latent, attn_weights, patches
        else:
            return global_latent

    def forward(self, image1, image2, return_attn=False):
        """
        Process two images and return their latent representations.
        If return_attn is True, return additional attention information for the first image.
        """
        if return_attn:
            latent1, attn1, patches = self.forward_image(image1, return_attn=True)
        else:
            latent1 = self.forward_image(image1, return_attn=False)
        latent2 = self.forward_image(image2, return_attn=False)
        if return_attn:
            return latent1, latent2, attn1, patches
        else:
            return latent1, latent2
