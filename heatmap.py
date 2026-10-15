"""
Visualization script for the Anomaly Autoencoder.

This script loads a single test image (e.g., from the MVTec AD dataset), passes it 
through the trained autoencoder, and generates a side-by-side comparison of the 
original image, the model's reconstruction, and an error heatmap. These visual 
diagnostics are highly effective for technical reports to demonstrate exactly 
where the model detects structural defects.
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# 1. IMPORT OUR MODEL
from model import AnomalyAutoencoder

def visualize_anomaly(model_path, image_path):
    """
    Loads a model and an image, computes the reconstruction error, 
    and displays a 3-panel figure showing the anomaly heatmap.
    """
    # Initialize the model
    model = AnomalyAutoencoder()
    # Ensure weights are mapped to CPU for visualization, allowing it to run on any machine
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Disable dropout/batchnorm for deterministic inference

    # Transformations
    # Resize to match the 256x256 dimensions expected by our encoder network
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Load image, ensure it has 3 color channels (RGB), and apply transforms
    img = Image.open(image_path).convert('RGB')
    
    # PyTorch expects batches of images: (Batch, Channels, Height, Width)
    # unsqueeze(0) adds a dummy batch dimension so the shape becomes (1, 3, 256, 256)
    input_tensor = transform(img).unsqueeze(0)

    # Network prediction
    with torch.no_grad(): # Disable gradient calculation for memory efficiency
        reconstruction = model(input_tensor)

    # Calculate the difference (heatmap)
    # Absolute pixel-wise difference between the original and the reconstructed image
    diff = torch.abs(input_tensor - reconstruction)
    
    # mean(dim=1) averages the error across the 3 RGB channels.
    # squeeze() removes the dummy batch dimension to get a 2D spatial map (256, 256).
    # .numpy() converts it from a PyTorch tensor to a standard NumPy array for plotting.
    heatmap = diff.mean(dim=1).squeeze().numpy()

    # Matplotlib expects images in (Height, Width, Channels) format.
    # permute(1, 2, 0) rearranges the PyTorch (C, H, W) format to match Matplotlib's requirement.
    orig_img = input_tensor.squeeze().permute(1, 2, 0).numpy()
    recon_img = reconstruction.squeeze().permute(1, 2, 0).numpy()

    # Plotting the charts
    # Create a 1x3 grid of subplots for side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original Input
    axes[0].imshow(orig_img)
    axes[0].set_title("Original (With anomaly)")
    axes[0].axis('off')
    
    # Panel 2: Model's Reconstruction (what the model thinks it *should* look like)
    axes[1].imshow(recon_img)
    axes[1].set_title("Network reconstruction")
    axes[1].axis('off')
    
    # Panel 3: Heatmap
    # cmap='jet' applies a color gradient where low errors are blue and high errors are red
    im = axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title("Anomaly Map (Heatmap)")
    axes[2].axis('off')
    
    # Add a color scale bar to quantify the error magnitude
    fig.colorbar(im, ax=axes[2])

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating a heatmap for a defective image.")
    # The default path to weights is now the one from the current folder
    parser.add_argument('--model_path', type=str, default='autoencoder_weights.pth', help='Path to model weights')
    # We expect someone to provide the path to the image via console
    parser.add_argument('--image_path', type=str, required=True, help='Path to the test image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at path {args.model_path}")
    elif not os.path.exists(args.image_path):
        print(f"Error: Image not found at path {args.image_path}")
    else:
        visualize_anomaly(args.model_path, args.image_path)