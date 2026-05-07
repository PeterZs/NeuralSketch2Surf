"""3D Swin Transformer blocks used by the S2V-Net backbone.

This standalone implementation keeps the windowed-attention structure of
SwinUNETR while exposing only the pieces needed for sparse 3D sketch inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils import ensure_tuple_rep
from ..layers import get_act_layer, get_norm_layer

class Mlp(nn.Module):
    """Feed-forward block inside each transformer layer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer="gelu", drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = get_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """Partition a 3D feature volume into local attention windows."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows

def window_reverse(windows, window_size, dims):
    """Reconstruct a 3D feature volume from local attention windows."""
    B = int(windows.shape[0] / (dims[0] * dims[1] * dims[2] / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, dims[0] // window_size[0], dims[1] // window_size[1], dims[2] // window_size[2], 
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, dims[0], dims[1], dims[2], -1)
    return x

class WindowAttention(nn.Module):
    """Window self-attention with discrete relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        stride_0 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        stride_1 = (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords[:, :, 0] * stride_0 + relative_coords[:, :, 1] * stride_1 + relative_coords[:, :, 2]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttentionV2(nn.Module):
    """Window self-attention with Swin V2 continuous position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0, 0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # Continuous position bias improves transfer across window sizes.
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(512, num_heads, bias=False)
        )

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_d = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w, relative_coords_d], indexing="ij"))
        
        relative_coords_table = relative_coords_table.permute(1, 2, 3, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2*Wd-1, 3
        
        if pretrained_window_size[0] > 0:
             relative_coords_table[:, :, :, :, 0] /= (pretrained_window_size[0] - 1)
             relative_coords_table[:, :, :, :, 1] /= (pretrained_window_size[1] - 1)
             relative_coords_table[:, :, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
             relative_coords_table[:, :, :, :, 0] /= (self.window_size[0] - 1)
             relative_coords_table[:, :, :, :, 1] /= (self.window_size[1] - 1)
             relative_coords_table[:, :, :, :, 2] /= (self.window_size[2] - 1)

        relative_coords_table *= 8  # Normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        stride_0 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        stride_1 = (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords[:, :, 0] * stride_0 + relative_coords[:, :, 1] * stride_1 + relative_coords[:, :, 2]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        
        max_log_val = torch.log(torch.tensor(1.0 / 0.01, device=self.logit_scale.device))
        logit_scale = torch.clamp(self.logit_scale, max=max_log_val).exp()
        
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Shifted-window transformer block for 3D feature tokens."""
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer="gelu", norm_layer="layer", use_v2=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_v2 = use_v2
        
        self.norm1 = get_norm_layer(norm_layer, 3, dim)
        
        if use_v2:
            self.attn = WindowAttentionV2(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity()
        self.norm2 = get_norm_layer(norm_layer, 3, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, (D, H, W))

        if any(i > 0 for i in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    """Downsample 3D tokens by merging neighboring patch features."""
    def __init__(self, dim, norm_layer="layer", spatial_dims=3):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = get_norm_layer(norm_layer, spatial_dims, 8 * dim)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
            
        x0 = x[:, 0::2, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1) 
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchMergingV2(nn.Module):
    """Swin V2 patch merging block."""
    def __init__(self, dim, norm_layer="layer", spatial_dims=3):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = get_norm_layer(norm_layer, spatial_dims, 8 * dim)

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (D % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
            
        x0 = x[:, 0::2, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1) 
        x = x.view(B, -1, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    """Stack shifted-window blocks at one feature resolution."""
    def __init__(self, dim, depth, num_heads, window_size, drop_path, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, norm_layer="layer", use_v2=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_v2=use_v2
            )
            for i in range(depth)
        ])

    def forward(self, x):
        H, W, D = self.input_resolution
        B, L, C = x.shape
        
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        
        x = x.view(B, D, H, W, C)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape
        x = x.view(B, -1, C)

        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
        d_slices = (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]), slice(-self.window_size[2], -self.shift_size[2]), slice(-self.shift_size[2], None))
        
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
                    
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.input_resolution = (Hp, Wp, Dp)
            x = blk(x, attn_mask)
        
        x = x.view(B, Dp, Hp, Wp, C)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        x = x.view(B, D * H * W, C)
            
        return x

class PatchEmbed(nn.Module):
    """Embed a voxel grid into non-overlapping 3D patch tokens."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, spatial_dims=3):
        super().__init__()
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = get_norm_layer(norm_layer, spatial_dims, embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, *x.shape[-3:])
        return x

class SwinTransformer(nn.Module):
    """Hierarchical 3D Swin Transformer encoder."""
    def __init__(self, in_chans=1, embed_dim=48, window_size=(7, 7, 7), patch_size=(2, 2, 2), depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer="layer", patch_norm=False, use_checkpoint=False, spatial_dims=3, downsample="merging", use_v2=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = ensure_tuple_rep(window_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.use_v2 = use_v2
        
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        
        self.downsamples = nn.ModuleList() 
        
        layer_classes = [self.layers1, self.layers2, self.layers3, self.layers4]
        
        if use_v2 or downsample == "mergingv2":
            DownsampleLayer = PatchMergingV2
        else:
            DownsampleLayer = PatchMerging

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                use_v2=use_v2
            )
            layer_classes[i_layer].append(layer)
            
            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    DownsampleLayer(dim=int(embed_dim * 2**i_layer), norm_layer=norm_layer, spatial_dims=spatial_dims)
                )

        self.num_features = int(embed_dim * 2**(self.num_layers - 1))

    def forward_features(self, x):
        x = self.patch_embed(x) 
        
        Wd, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
        
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
        outs = []
        
        self.layers1[0].input_resolution = (Wh, Ww, Wd)
        x = self.layers1[0](x)
        outs.append(x)
        
        self.downsamples[0].input_resolution = (Wh, Ww, Wd)
        x = self.downsamples[0](x)
        Wh, Ww, Wd = (Wh + 1) // 2, (Ww + 1) // 2, (Wd + 1) // 2
        
        self.layers2[0].input_resolution = (Wh, Ww, Wd)
        x = self.layers2[0](x)
        outs.append(x)
        
        self.downsamples[1].input_resolution = (Wh, Ww, Wd)
        x = self.downsamples[1](x)
        Wh, Ww, Wd = (Wh + 1) // 2, (Ww + 1) // 2, (Wd + 1) // 2
        
        self.layers3[0].input_resolution = (Wh, Ww, Wd)
        x = self.layers3[0](x)
        outs.append(x)
        
        self.downsamples[2].input_resolution = (Wh, Ww, Wd)
        x = self.downsamples[2](x)
        Wh, Ww, Wd = (Wh + 1) // 2, (Ww + 1) // 2, (Wd + 1) // 2
        
        self.layers4[0].input_resolution = (Wh, Ww, Wd)
        x = self.layers4[0](x)
        outs.append(x)
        
        return outs
