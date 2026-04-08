from typing import Optional

import torch
import torch.nn as nn

from .layers import DropPath
from .vit import Mlp
from .vit_chunk import ChunkedAttention
from .vit_chunk_base import ChunkedVisionTransformerBase


class ChunkedBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
    ) -> None:
        super().__init__()
        if int(mlp_ratio) != mlp_ratio:
            raise ValueError("mlp_ratio must be an integer for chunked MLP")
        self.norm1 = norm_layer(dim)
        self.attn = ChunkedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(mlp_ratio) * dim,
            act_layer=act_layer,
            drop=drop,
        )
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qkv_3cc, proj_1cc = torch.split(weights, [3, 1], dim=0)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), qkv_3cc, proj_1cc))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), qkv_3cc, proj_1cc))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ChunkedAttenVisionTransformer(ChunkedVisionTransformerBase):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
    ) -> None:
        super().__init__(
            block_cls=ChunkedBlock,
            num_matrix=4,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )
