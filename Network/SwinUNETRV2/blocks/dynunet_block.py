import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers import get_norm_layer, get_act_layer
from ..utils import ensure_tuple_rep

class UnetBasicBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name="leakyrelu"):
        super().__init__()
        kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        stride = ensure_tuple_rep(stride, spatial_dims)
        padding = [(k - 1) // 2 for k in kernel_size]
        
        if spatial_dims == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        elif spatial_dims == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            
        self.norm = get_norm_layer(norm_name, spatial_dims, out_channels)
        self.lrelu = get_act_layer(act_name)

    def forward(self, x):
        return self.lrelu(self.norm(self.conv(x)))

class UnetResBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name="leakyrelu"):
        super().__init__()
        kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        stride = ensure_tuple_rep(stride, spatial_dims)
        
        self.conv1 = UnetBasicBlock(spatial_dims, in_channels, out_channels, kernel_size, stride, norm_name, act_name)
        self.conv2 = UnetBasicBlock(spatial_dims, out_channels, out_channels, kernel_size, 1, norm_name, act_name)
        
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            if spatial_dims == 3:
                self.downsample = nn.Conv3d(in_channels, out_channels, 1, stride)
            else:
                self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride)
            self.norm_ds = get_norm_layer(norm_name, spatial_dims, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.norm_ds(self.downsample(x))
            
        out = out + residual
        return out

class UnetUpBlock(nn.Module):
    def __init__(
        self, 
        spatial_dims, 
        in_channels, 
        out_channels, 
        skip_channels, 
        kernel_size, 
        stride, 
        norm_name, 
        act_name="leakyrelu",
        upsample_kernel_size=None
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        
        self.conv_adjust = UnetBasicBlock(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_name=norm_name,
            act_name=act_name
        )

        conv_in_channels = out_channels + skip_channels
        
        self.conv_block = UnetResBlock(
            spatial_dims, 
            conv_in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            norm_name=norm_name, 
            act_name=act_name
        )

    def forward(self, x, skip):
        # Trilinear interpolation(replacing transpose convolution)
        if self.spatial_dims == 3:
            x_up = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        else:
            x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # adjust channel dimensions
        out = self.conv_adjust(x_up)
        
        # concatenate skip connection
        out = torch.cat((out, skip), dim=1)
        
        # Convolution Fusion
        out = self.conv_block(out)
        return out

class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, dropout=None):
        super().__init__()
        if spatial_dims == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        return self.conv(x)
