import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder designed for Anomaly Detection.
    Expects 3-channel (RGB) input images scaled between [0, 1].
    """
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        
        # ---------------------------------------------------------
        # ENCODER: Compresses spatial dimensions and expands channels
        # Total spatial downsampling factor: 8 (2 * 2 * 2)
        # ---------------------------------------------------------
        self.encoder = nn.Sequential(
            # Input: (B, 3, H, W) -> Output: (B, 16, H/2, W/2)
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.ReLU(),
            
            # Input: (B, 16, H/2, W/2) -> Output: (B, 32, H/4, W/4)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            
            # Input: (B, 32, H/4, W/4) -> Output: (B, 64, H/8, W/8)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.ReLU()
        )
        
        # ---------------------------------------------------------
        # DECODER: Reconstructs spatial dimensions and compresses channels
        # ---------------------------------------------------------
        self.decoder = nn.Sequential(
            # Input: (B, 64, H/8, W/8) -> Output: (B, 32, H/4, W/4)
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            
            # Input: (B, 32, H/4, W/4) -> Output: (B, 16, H/2, W/2)
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            
            # Input: (B, 16, H/2, W/2) -> Output: (B, 3, H, W)
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), 
            # Sigmoid ensures the reconstructed pixels remain in the [0, 1] range
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the autoencoder.
        
        Args:
            x (torch.Tensor): Input batch of images of shape (B, 3, H, W)
            
        Returns:
            torch.Tensor: Reconstructed images of shape (B, 3, H, W)
        """
        return self.decoder(self.encoder(x))