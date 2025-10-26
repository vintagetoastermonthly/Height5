import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict, List


class MultiViewEncoder(nn.Module):
    """Encoder that processes 5 views through ResNet50 backbones."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Create 5 ResNet50 encoders (one per view)
        self.encoders = nn.ModuleDict({
            'ortho': self._create_resnet_encoder(pretrained),
            'north': self._create_resnet_encoder(pretrained),
            'south': self._create_resnet_encoder(pretrained),
            'east': self._create_resnet_encoder(pretrained),
            'west': self._create_resnet_encoder(pretrained),
        })

        # Reduction convs: 1024 -> 64 channels
        self.reduce_layer3 = nn.Conv2d(1024, 64, kernel_size=1)
        self.reduce_layer2 = nn.Conv2d(512, 64, kernel_size=1)

        # Downsample from 32x32 to 8x8
        self.downsample = nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1)

    def _create_resnet_encoder(self, pretrained: bool):
        """Create ResNet50 encoder and extract intermediate layers."""
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = resnet50(weights=weights)

        # We'll manually extract features, so we need the layers
        encoder = nn.ModuleDict({
            'conv1': resnet.conv1,
            'bn1': resnet.bn1,
            'relu': resnet.relu,
            'maxpool': resnet.maxpool,
            'layer1': resnet.layer1,
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
        })

        return encoder

    def forward_single(self, x: torch.Tensor, view: str) -> Dict[str, torch.Tensor]:
        """Forward pass for a single view, return layer2 and layer3 features."""
        encoder = self.encoders[view]

        x = encoder['conv1'](x)
        x = encoder['bn1'](x)
        x = encoder['relu'](x)
        x = encoder['maxpool'](x)

        x = encoder['layer1'](x)
        layer2_feat = encoder['layer2'](x)  # (B, 512, 32, 32)
        layer3_feat = encoder['layer3'](layer2_feat)  # (B, 1024, 16, 16)

        return {
            'layer2': layer2_feat,
            'layer3': layer3_feat
        }

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: Dict with keys 'ortho', 'north', 'south', 'east', 'west'
                    Each value is (B, 3, 256, 256)

        Returns:
            Dict with:
                - 'encoded_features': List of 5 tensors (B, 64, 4, 4)
                - 'ortho_skip': (B, 64, 16, 16) skip connection from ortho layer2
        """
        # Process all views
        all_features = []
        ortho_layer2 = None

        for view in ['ortho', 'north', 'south', 'east', 'west']:
            feats = self.forward_single(images[view], view)

            # Extract and reduce layer3 features
            layer3 = self.reduce_layer3(feats['layer3'])  # (B, 64, 16, 16)
            layer3 = self.downsample(layer3)  # (B, 64, 4, 4)
            all_features.append(layer3)

            # Save ortho layer2 for skip connection
            if view == 'ortho':
                ortho_layer2 = feats['layer2']

        # Process ortho skip connection
        ortho_skip = self.reduce_layer2(ortho_layer2)  # (B, 64, 32, 32)
        ortho_skip = F.avg_pool2d(ortho_skip, kernel_size=2)  # (B, 64, 16, 16)

        return {
            'encoded_features': all_features,  # List of 5 × (B, 64, 4, 4)
            'ortho_skip': ortho_skip  # (B, 64, 16, 16)
        }


class CrossAttentionFusion(nn.Module):
    """Cross-attention module to fuse multi-view features."""

    def __init__(self, feature_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: List of 5 tensors, each (B, 64, 4, 4)
                          [ortho, north, south, east, west]

        Returns:
            Fused features: (B, 64, 4, 4)
        """
        B = features_list[0].shape[0]
        _, C, H, W = features_list[0].shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        flattened = []
        for feat in features_list:
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            flattened.append(feat_flat)

        # Ortho as query, all views as key/value
        ortho_query = flattened[0]  # (B, 16, 64)
        all_kv = torch.cat(flattened, dim=1)  # (B, 80, 64) where 80 = 5*16

        # Cross-attention
        fused, _ = self.cross_attention(
            query=ortho_query,
            key=all_kv,
            value=all_kv
        )  # (B, 16, 64)

        # Residual connection
        fused = self.norm(fused + ortho_query)

        # Reshape back to spatial: (B, 16, 64) -> (B, 64, 4, 4)
        fused = fused.permute(0, 2, 1).reshape(B, self.feature_dim, H, W)

        return fused


class SubPixelUpsample(nn.Module):
    """Upsample using sub-pixel convolution."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    """Decoder with skip connections and sub-pixel upsampling."""

    def __init__(self):
        super().__init__()

        # First upsample: 4x4x64 -> 8x8x128
        self.upsample1 = SubPixelUpsample(64, 128, scale_factor=2)

        # Second upsample: 8x8x128 -> 16x16x64
        self.upsample2 = SubPixelUpsample(128, 64, scale_factor=2)

        # Conv block after skip connection concatenation
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 64 + 64 skip = 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Third upsample: 16x16x64 -> 32x32x32
        self.upsample3 = SubPixelUpsample(64, 32, scale_factor=2)

        # Final conv to output
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, fused: torch.Tensor, ortho_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, 64, 4, 4) fused features
            ortho_skip: (B, 64, 16, 16) skip connection from ortho

        Returns:
            output: (B, 1, 32, 32) height map
        """
        # Upsample to 8x8
        x = self.upsample1(fused)  # (B, 128, 8, 8)

        # Upsample to 16x16
        x = self.upsample2(x)  # (B, 64, 16, 16)

        # Concatenate with skip connection
        x = torch.cat([x, ortho_skip], dim=1)  # (B, 128, 16, 16)

        # Process with conv block
        x = self.conv_block1(x)  # (B, 64, 16, 16)

        # Upsample to 32x32
        x = self.upsample3(x)  # (B, 32, 32, 32)

        # Final conv to height map
        output = self.final_conv(x)  # (B, 1, 32, 32)

        return output


class HeightEstimationModel(nn.Module):
    """Complete multi-view height estimation model."""

    def __init__(self, pretrained: bool = True):
        super().__init__()

        self.encoder = MultiViewEncoder(pretrained=pretrained)
        self.fusion = CrossAttentionFusion(feature_dim=64, num_heads=8)
        self.decoder = Decoder()

    def forward(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: Dict with keys 'ortho', 'north', 'south', 'east', 'west'
                    Each value is (B, 3, 256, 256)

        Returns:
            height_map: (B, 1, 32, 32) predicted height map
        """
        # Encode all views
        encoder_out = self.encoder(images)
        encoded_features = encoder_out['encoded_features']
        ortho_skip = encoder_out['ortho_skip']

        # Fuse features with cross-attention
        fused = self.fusion(encoded_features)

        # Decode to height map
        height_map = self.decoder(fused, ortho_skip)

        return height_map


if __name__ == '__main__':
    # Test model
    model = HeightEstimationModel(pretrained=False)
    batch_size = 2

    # Create dummy inputs
    dummy_images = {
        'ortho': torch.randn(batch_size, 3, 256, 256),
        'north': torch.randn(batch_size, 3, 256, 256),
        'south': torch.randn(batch_size, 3, 256, 256),
        'east': torch.randn(batch_size, 3, 256, 256),
        'west': torch.randn(batch_size, 3, 256, 256),
    }

    # Forward pass
    output = model(dummy_images)
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 1, 32, 32)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
