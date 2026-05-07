"""Small layer factory helpers used by the standalone SwinUNETR modules."""
import torch.nn as nn

def get_norm_layer(name, spatial_dims=3, channels=None):
    """Return the normalization layer requested by the network config."""
    if name == "layer":
        return nn.LayerNorm(normalized_shape=channels)
    
    if name == "instance":
        if spatial_dims == 3:
            return nn.InstanceNorm3d(channels, affine=True)
        elif spatial_dims == 2:
            return nn.InstanceNorm2d(channels, affine=True)
            
    if name == "batch":
        if spatial_dims == 3:
            return nn.BatchNorm3d(channels)
        elif spatial_dims == 2:
            return nn.BatchNorm2d(channels)
            
    raise NotImplementedError(f"Norm type {name} not implemented in standalone version")

def get_act_layer(name):
    """Return the activation layer requested by the network config."""
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "prelu":
        return nn.PReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
        
    raise NotImplementedError(f"Activation {name} not implemented in standalone version")
