"""
Evaluation script for the Anomaly Autoencoder.

This script loads trained weights and evaluates the model on a test dataset 
containing both healthy and anomalous images. It uses a dual-thresholding 
approach (pixel-wise difference and anomalous area size) to classify images 
and generates a performance report with standard classification metrics.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os
import argparse

# Import model architecture from external model.py module
from model import AnomalyAutoencoder

# TESTING LOGIC
def evaluate_model(model_path, test_dir, pixel_threshold=0.15, min_area=50):
    """
    Evaluates the autoencoder on the test set using specific detection thresholds.
    
    Args:
        model_path (str): Path to the saved .pth weights file.
        test_dir (str): Path to the testing dataset directory.
        pixel_threshold (float): Minimum absolute difference [0-1] for a pixel to be considered anomalous.
        min_area (int): Minimum number of anomalous pixels required to flag the entire image as an anomaly.
    """
    print("Loading model and data...")
    
    # Initialize the model and load weights on CPU
    model = AnomalyAutoencoder()
    # map_location='cpu' ensures we can evaluate the model even if it was trained on a GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # .eval() disables training-specific layers like Dropout and Batch Normalization during inference
    model.eval()

    # Load test data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    # Batch size = 1, so we can analyze images individually
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Dataset automatically maps folder names to numbers (e.g., 'good': 0, 'scratch': 1)
    # Save the index of the 'good' folder to know which images are truly healthy
    good_class_idx = test_dataset.class_to_idx['good']

    y_true = [] # True labels (0 = Healthy, 1 = Anomaly)
    y_pred = [] # Labels predicted by our network

    print("Starting image scanning. This may take several seconds...\n")

    # torch.no_grad() disables gradient tracking to save memory and significantly speed up computation
    with torch.no_grad():
        for images, labels in test_loader:
            # Determine the true state (Ground Truth)
            # If the image is from the 'good' folder, mark as 0. Otherwise 1.
            is_anomaly_true = 0 if labels.item() == good_class_idx else 1
            y_true.append(is_anomaly_true)

            # Network prediction
            reconstruction = model(images)
            
            # 1. Reconstruction error on each pixel (Mean Absolute Error per pixel)
            diff = torch.abs(images - reconstruction)
            
            # 2. Pixel Threshold: Create a mask of pixels that differ by more than the set threshold
            # This creates a boolean tensor where 'True' represents a defective pixel
            anomaly_mask = diff > pixel_threshold
            
            # 3. Area Threshold: Count all active (defective) pixels from the mask
            anomalous_pixels_count = torch.sum(anomaly_mask).item()
            
            # If the error area is larger than the minimum, the model declares an anomaly
            if anomalous_pixels_count > min_area:
                y_pred.append(1)
            else:
                y_pred.append(0)

    # CALCULATING METRICS FOR THE REPORT
    # Using scikit-learn to compare ground truth against model predictions
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("-" * 30)
    print("EVALUATION RESULTS:")
    print("-" * 30)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision:  {prec*100:.2f}%")
    print(f"Recall:      {rec*100:.2f}%")
    print("\nConfusion Matrix:")
    # cm structure: [[True Neg, False Pos], [False Neg, True Pos]]
    print(f"True Negatives  (Healthy predicted as healthy):    {cm[0][0]}")
    print(f"False Positives (Healthy predicted as defective):   {cm[0][1]}")
    print(f"False Negatives (Defective predicted as healthy):   {cm[1][0]}")
    print(f"True Positives  (Defective predicted as defective):  {cm[1][1]}")
    print("-" * 30)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of the Autoencoder on the test set.")
    
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test folder')
    parser.add_argument('--model_path', type=str, default='autoencoder_weights.pth', help='Path to the model weights')
    
    # nargs='+' allows passing multiple values from the terminal (e.g., --pixel_threshold 0.1 0.15 0.2)
    # This acts as an automated grid search for finding the best hyperparameters
    parser.add_argument('--pixel_threshold', type=float, nargs='+', default=[0.15], help='Pixel error thresholds (e.g., 0.12 0.15 0.18)')
    parser.add_argument('--min_area', type=int, nargs='+', default=[20], help='Minimum anomaly areas (e.g., 10 20 50)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Data folder not found: {args.test_dir}")
    elif not os.path.exists(args.model_path):
        print(f"Error: Model weights not found: {args.model_path}")
    else:
        # Nested loops iterate through all combinations of provided thresholds and areas
        for pt in args.pixel_threshold:
            for ma in args.min_area:
                print(f"\n>>> TESTING PARAMETERS: Pixel Threshold = {pt}, Min Area = {ma} <<<")
                evaluate_model(
                    args.model_path, 
                    args.test_dir, 
                    pixel_threshold=pt, 
                    min_area=ma
                )