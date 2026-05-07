"""Local residual refinement network for S2V-Net occupancy logits."""
import torch
import torch.nn as nn

class RefinementBlock(nn.Module):
    """3D residual block."""
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
    """Refine coarse logits with local residual corrections."""
    def __init__(self, in_channels=1, hidden_dim=32, num_blocks=3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.body = nn.Sequential(
            *[RefinementBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        self.tail = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        correction = self.tail(feat)
        # Residual prediction preserves the backbone topology while correcting local artifacts.
        return x + correction
