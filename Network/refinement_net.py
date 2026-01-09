import torch
import torch.nn as nn

class RefinementBlock(nn.Module):
    """标准的 3D 残差块"""
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
        
        # 1. 升维提取特征
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. 堆叠残差块 (去噪核心)
        self.body = nn.Sequential(
            *[RefinementBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        # 3. 降维预测修正值
        self.tail = nn.Conv3d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: 来自 SwinUNETR 的粗糙 logits
        feat = self.head(x)
        feat = self.body(feat)
        correction = self.tail(feat)
        
        # 输出 = 原始粗糙结果 + 修正值
        return x + correction