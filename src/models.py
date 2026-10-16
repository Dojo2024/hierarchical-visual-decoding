"""Neural architectures for Hierarchical Visual Cortex Decoding on NSD.
Includes:
  - SingleStreamMLP: baseline operating on full cortex, early visual, or higher visual stream alone.
  - HierarchicalDualStreamMLP: disentangled dual-stream network with adaptive cross-stream gated fusion.
  - HierarchicalCLIPRegressor: regresses continuous 768-d CLIP visual latents from hierarchical cortical streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 24
CLIP_DIM = 768


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, device=x.device) < keep_prob
        return x * mask / keep_prob


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.5, drop_path: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.block(x))


class SingleStreamMLP(nn.Module):
    """Standard ResidualMLP baseline operating on a single cortical stream (full, early, or higher)."""

    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.head(x)


class HierarchicalDualStreamMLP(nn.Module):
    """Disentangled dual-stream network:
    - Stream 1: Early Visual Cortex (retinotopic, spatial structure)
    - Stream 2: Higher Visual Cortex (ventral stream, invariant semantics)
    - Fusion: Adaptive cross-stream gating mechanism
    """

    def __init__(
        self,
        dim_early: int = 96,
        dim_higher: int = 96,
        latent_dim: int = 128,
        depth: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        drop_path: float = 0.1,
    ):
        super().__init__()
        # Initial projection to common latent dimension
        self.proj_early = nn.Linear(dim_early, latent_dim)
        self.proj_higher = nn.Linear(dim_higher, latent_dim)

        # Independent stream residual processors
        self.stream_early = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.stream_higher = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )

        # Dynamic cross-stream gating
        self.gate_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        # Integrated cross-stream fusion blocks
        self.fusion_blocks = nn.Sequential(
            ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path),
            ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x_early: torch.Tensor, x_higher: torch.Tensor) -> torch.Tensor:
        h_early = self.stream_early(self.proj_early(x_early))
        h_higher = self.stream_higher(self.proj_higher(x_higher))

        # Compute per-feature dynamic gating weights
        gate = self.gate_fc(torch.cat([h_early, h_higher], dim=-1))
        fused = gate * h_early + (1.0 - gate) * h_higher

        fused = self.fusion_blocks(fused)
        return self.head(fused)


class HierarchicalCLIPRegressor(nn.Module):
    """Regresses continuous 768-d CLIP visual embeddings from dual cortical streams."""

    def __init__(
        self,
        dim_early: int = 96,
        dim_higher: int = 96,
        latent_dim: int = 256,
        depth: int = 4,
        clip_dim: int = CLIP_DIM,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.proj_early = nn.Linear(dim_early, latent_dim)
        self.proj_higher = nn.Linear(dim_higher, latent_dim)

        self.stream_early = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.stream_higher = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )

        self.gate_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path),
            ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path),
        )

        self.clip_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, clip_dim),
        )

    def forward(self, x_early: torch.Tensor, x_higher: torch.Tensor) -> torch.Tensor:
        h_e = self.stream_early(self.proj_early(x_early))
        h_h = self.stream_higher(self.proj_higher(x_higher))
        gate = self.gate_fc(torch.cat([h_e, h_h], dim=-1))
        fused = self.fusion(gate * h_e + (1.0 - gate) * h_h)
        pred_clip = self.clip_head(fused)
        return F.normalize(pred_clip, p=2, dim=-1)


class SingleStreamCLIPRegressor(nn.Module):
    """Regresses continuous 768-d CLIP visual embeddings from a single cortical stream (full, early, or higher)."""

    def __init__(
        self,
        dim: int = 96,
        latent_dim: int = 256,
        depth: int = 4,
        clip_dim: int = CLIP_DIM,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(dim, latent_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.clip_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, clip_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.proj(x))
        pred_clip = self.clip_head(h)
        return F.normalize(pred_clip, p=2, dim=-1)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class Flatmap2DConvNet(nn.Module):
    """2D Spatial ConvNet operating directly on the 215x200 cortical flatmap canvas.
    Preserves spatial cortical topography and retinotopic neighborhood structure.
    Has ~113k parameters to directly match the 1D Monolithic MLP baseline (115k).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_ch: int = 80, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )
        self.stage1 = DepthwiseSeparableConv2d(base_ch, int(base_ch * 1.5), stride=2)       # 120
        self.stage2 = DepthwiseSeparableConv2d(int(base_ch * 1.5), base_ch * 3, stride=2)   # 240
        self.stage3 = DepthwiseSeparableConv2d(base_ch * 3, base_ch * 3, stride=2)          # 240
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(base_ch * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 215, 200)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)
