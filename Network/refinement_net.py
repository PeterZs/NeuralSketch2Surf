import torch
import torch.nn as nn

class RefinementBlock(nn.Module):
    """3D Residual Block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual

class RefinementNet(nn.Module):
    """
    RefinementNet：Coarse Logits -> Fine Logits
    """
    def __init__(self, in_channels=1, hidden_dim=32, num_blocks=3):
        super().__init__()
        
        # Feature extraction through dimensionality expansion
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Stacked residual blocks
        self.body = nn.Sequential(
            *[RefinementBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        # Dimension Reduction Prediction Correction Value
        self.tail = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: coarse logits from SwinUNETR
        feat = self.head(x)
        feat = self.body(feat)
        correction = self.tail(feat)
        
        # Output = coarse logits + Correction value

        return x + correction
