"""
Training script for the Anomaly Autoencoder.

This script loads a dataset of 'healthy' baseline images, trains the model to 
reconstruct them using Mean Squared Error (MSE), saves the trained weights, 
and generates a learning curve plot for visual validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

# 1. IMPORT OUR MODEL FROM THE model.py FILE
from model import AnomalyAutoencoder

# 2. DATA LOADING
def get_dataloader(data_dir, batch_size=16):
    """
    Creates a PyTorch DataLoader for the training images.
    
    Args:
        data_dir (str): Path to the dataset directory. 
        batch_size (int): Number of images processed in a single forward pass.
        
    Returns:
        DataLoader: An iterable over the dataset.
    """
    # Define the preprocessing pipeline: resize to uniform dimensions and convert to tensor
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # ImageFolder expects images to be inside a subdirectory (which aligns perfectly 
    # with how the MVTec AD dataset separates classes like 'wood/train/good/')
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. TRAINING LOOP
def train_model(data_dir, save_path, num_epochs):
    """
    Executes the training loop, saving the model weights and loss plot upon completion.
    """
    dataloader = get_dataloader(data_dir)
    model = AnomalyAutoencoder()
    
    #Line wrote by gemini
    # Dynamically select the best hardware accelerator available
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting training on device: {device}\n")
    model.to(device)

    # MSELoss penalizes large pixel differences, forcing the model to learn a good generalized reconstruction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    history_loss = [] 

    for epoch in range(num_epochs):
        model.train() # Set model to training mode (enables gradients, dropout, etc.)
        running_loss = 0.0

        # Iterate over batches of images (the '_' ignores the labels, since autoencoders are unsupervised)
        for images, _ in dataloader:
            images = images.to(device)
            
            # Step 1: Clear old gradients from the previous step
            optimizer.zero_grad()
            
            # Step 2: Forward pass (compress and reconstruct)
            outputs = model(images)
            
            # Step 3: Calculate how far off the reconstruction is from the original
            loss = criterion(outputs, images) 
            
            # Step 4: Backward pass (calculate gradients)
            loss.backward()
            
            # Step 5: Update the model's weights based on the gradients
            optimizer.step()
            
            # Accumulate the total loss for this batch
            running_loss += loss.item() * images.size(0)

        # Calculate the average loss across the entire epoch
        epoch_loss = running_loss / len(dataloader.dataset)
        history_loss.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f}")

    # Saving weights
    # state_dict() saves just the learned parameters, which is the PyTorch standard practice
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights have been saved to the file: {save_path}")

    # Generating chart
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), history_loss, marker='o', color='b', label='Training Loss (MSE)')
    plt.title('Learning Curve - Convolutional Autoencoder')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.savefig("learning_curve.png")

# 4. MAIN CALL WITH ARGPARSE
if __name__ == "__main__":
    # Setup command-line arguments for easier execution from the terminal
    parser = argparse.ArgumentParser(description="Training the Autoencoder on healthy samples.")
    parser.add_argument('--data_dir', type=str, default='./wood/train', help='Path to the training data folder')
    parser.add_argument('--save_path', type=str, default='autoencoder_weights.pth', help='File name to save the weights')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Validate the data directory exists before starting the expensive training process
    if not os.path.exists(args.data_dir):
        print(f"Error: Data folder '{args.data_dir}' not found. Check the path.")
    else:
        train_model(args.data_dir, args.save_path, args.epochs)