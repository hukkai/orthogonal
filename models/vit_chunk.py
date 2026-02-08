import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DropPath, trunc_normal_
from .vit import PatchEmbed


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
        qkv_w_3cc: torch.Tensor,
        proj_w_1cc: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv_w = qkv_w_3cc.reshape(3 * C, C)
        qkv = F.linear(x, qkv_w, bias=self.qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        proj_w = proj_w_1cc.squeeze(0)
        x = F.linear(x, proj_w, bias=self.proj_bias)
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

    def forward(self, x: torch.Tensor, w1_rcc: torch.Tensor, w2_rcc: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        w1 = w1_rcc.reshape(self.r * C, C)
        w2 = w2_rcc.permute(1, 0, 2).reshape(C, self.r * C)

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

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        qkv_3cc, proj_1cc, w1_rcc, w2_rcc = torch.split(
            weights, [3, 1, self.mlp.r, self.mlp.r], dim=0
        )
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), qkv_3cc, proj_1cc))
            x = x + self.drop_path(self.mlp(self.norm2(x), w1_rcc, w2_rcc))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), qkv_3cc, proj_1cc))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), w1_rcc, w2_rcc))
        return x


class ChunkedVisionTransformer(nn.Module):
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
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                ChunkedBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.num_matrix = 4 + 2 * int(mlp_ratio)
        total_chunks = depth * self.num_matrix
        self.chunk_weights = nn.Parameter(
            torch.randn(total_chunks, embed_dim, embed_dim) / math.sqrt(embed_dim)
        )

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        with torch.no_grad():
            w = self.chunk_weights.data.to(dtype=torch.float64)
            q, r = torch.linalg.qr(w)
            diag = torch.diagonal(r, dim1=-2, dim2=-1)
            sign = torch.sign(diag)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            q = q * sign.unsqueeze(-2)
            sign_det, _ = torch.linalg.slogdet(w)
            mask = sign_det < 0
            if mask.any():
                q[mask, :, -1] *= -1
            self.chunk_weights.data.copy_(q.to(dtype=self.chunk_weights.dtype))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        num_patches = x.shape[1] - 1
        if num_patches == self.pos_embed.shape[1] - 1:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        pos = self.pos_embed[:, 1:]
        dim = pos.shape[-1]
        h0 = self.patch_embed.grid_size[0]
        w0 = self.patch_embed.grid_size[1]
        pos = pos.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        return torch.cat((cls_pos, pos), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        block_w = self.chunk_weights.reshape(
            len(self.blocks), self.num_matrix, self.embed_dim, self.embed_dim
        )

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self._interpolate_pos_encoding(
            x, H // self.patch_embed.patch_size, W // self.patch_embed.patch_size
        )
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk, w in zip(self.blocks, block_w):
            x = blk(x, w)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x