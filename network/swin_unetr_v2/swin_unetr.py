"""Modified 3D SwinUNETR backbone for sketch-to-voxel prediction."""
import torch
import torch.nn as nn

from .utils import look_up_option, ensure_tuple_rep
from .blocks.swin_transformer import SwinTransformer
from .blocks.dynunet_block import UnetOutBlock, UnetResBlock, UnetUpBlock
from .layers import get_norm_layer

class SwinUNETR(nn.Module):
    """3D SwinUNETR backbone adapted for sparse sketch occupancy."""

    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=True,
    ):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 3):
             raise ValueError("Standalone SwinUNETR only supports 3D currently.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        if use_v2:
            downsample = "mergingv2"
            
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer="layer",
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, ["merging", "mergingv2"], default="merging"),
            use_v2=use_v2
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.encoder2 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.encoder3 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.encoder4 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.encoder10 = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder5 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            skip_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder4 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            skip_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder3 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            skip_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.decoder2 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            skip_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in):
        """Return coarse occupancy logits from a sparse sketch volume."""
        D, H, W = x_in.shape[2], x_in.shape[3], x_in.shape[4]
        
        hidden_states_out = self.swinViT.forward_features(x_in)
        
        def project(x, d, h, w):
            """Restore token sequences to 3D feature maps for skip fusion."""
            B, L, C = x.shape
            x = x.view(B, d, h, w, C)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            return x

        enc0 = self.encoder1(x_in)

        d1, h1, w1 = D // 2, H // 2, W // 2
        enc1 = self.encoder2(project(hidden_states_out[0], d1, h1, w1))

        d2, h2, w2 = d1 // 2, h1 // 2, w1 // 2
        enc2 = self.encoder3(project(hidden_states_out[1], d2, h2, w2))

        d3, h3, w3 = d2 // 2, h2 // 2, w2 // 2
        enc3 = self.encoder4(project(hidden_states_out[2], d3, h3, w3))

        d4, h4, w4 = d3 // 2, h3 // 2, w3 // 2
        dec4 = self.encoder10(project(hidden_states_out[3], d4, h4, w4))

        dec3 = self.decoder5(dec4, enc3) 
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        out = self.decoder2(dec1, enc0)
        
        return self.out(out)
