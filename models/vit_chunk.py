from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DropPath
from .vit_chunk_base import ChunkedVisionTransformerBase


class ChunkedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_bias = nn.Parameter(torch.zeros(dim * 3)) if qkv_bias else None
        self.proj_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        qkv_weight: torch.Tensor,
        proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(x, qkv_weight, bias=self.qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = F.linear(x, proj_weight, bias=self.proj_bias)
        x = self.proj_drop(x)
        return x


class ChunkedMlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.r = int(mlp_ratio)
        if self.r != mlp_ratio:
            raise ValueError("mlp_ratio must be an integer for chunked MLP")
        self.hidden_dim = dim * self.r
        self.bias1 = nn.Parameter(torch.zeros(self.hidden_dim)) if bias else None
        self.bias2 = nn.Parameter(torch.zeros(dim)) if bias else None
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, w1, bias=self.bias1)
        x = self.act(x)
        x = self.drop(x)
        x = F.linear(x, w2, bias=self.bias2)
        x = self.drop(x)
        return x


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
        self.mlp = ChunkedMlp(dim, mlp_ratio=mlp_ratio, bias=qkv_bias, drop=drop, act_layer=act_layer)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
        qkv_weight = torch.cat((weights["q"], weights["k"], weights["v"]), dim=0)
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), qkv_weight, weights["proj"]))
            x = x + self.drop_path(self.mlp(self.norm2(x), weights["w1"], weights["w2"]))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), qkv_weight, weights["proj"]))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), weights["w1"], weights["w2"]))
        return x


class ChunkedVisionTransformer(ChunkedVisionTransformerBase):
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
        orth_dim: int | None = None,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: Optional[float] = None,
    ) -> None:
        orth_dim = embed_dim if orth_dim is None else orth_dim
        super().__init__(
            block_cls=ChunkedBlock,
            weight_names=("q", "k", "v", "proj", "w1", "w2"),
            orth_dim=orth_dim,
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
