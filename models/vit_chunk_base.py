from dataclasses import dataclass
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import trunc_normal_
from .vit import PatchEmbed


@dataclass(frozen=True)
class WeightSpec:
    name: str
    num_blocks: int


def blocks_to_matrix(blocks: torch.Tensor) -> torch.Tensor:
    if blocks.ndim != 3:
        raise ValueError("blocks must have shape (num_blocks, dim, orth_dim)")
    return blocks.permute(1, 0, 2).reshape(blocks.shape[1], blocks.shape[0] * blocks.shape[2])


class ChunkedVisionTransformerBase(nn.Module):
    def __init__(
        self,
        *,
        block_cls: Type[nn.Module],
        weight_names: tuple[str, ...],
        orth_dim: int,
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
        self.orth_dim = orth_dim

        if orth_dim <= 0 or embed_dim % orth_dim != 0:
            raise ValueError("orth_dim must be a positive divisor of embed_dim")

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
                block_cls(
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

        self.weight_specs = self._build_weight_specs(weight_names, embed_dim, orth_dim, mlp_ratio)
        self.blocks_per_layer = sum(spec.num_blocks for spec in self.weight_specs)
        total_chunks = depth * self.blocks_per_layer
        self.chunk_weights = nn.Parameter(
            torch.randn(total_chunks, embed_dim, orth_dim)
        )

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        with torch.no_grad():
            w = self.chunk_weights.data.to(dtype=torch.float64)
            q, _ = torch.linalg.qr(w, mode="reduced")
            self.chunk_weights.data.copy_(q.to(dtype=self.chunk_weights.dtype))

    @staticmethod
    def _build_weight_specs(
        weight_names: tuple[str, ...],
        embed_dim: int,
        orth_dim: int,
        mlp_ratio: float,
    ) -> tuple[WeightSpec, ...]:
        attn_blocks = embed_dim // orth_dim
        specs: list[WeightSpec] = []
        mlp_expansion = None

        for name in weight_names:
            if name in {"q", "k", "v", "proj"}:
                specs.append(WeightSpec(name=name, num_blocks=attn_blocks))
                continue
            if name in {"w1", "w2"}:
                if mlp_expansion is None:
                    mlp_expansion = int(mlp_ratio)
                    if mlp_expansion != mlp_ratio:
                        raise ValueError("mlp_ratio must be an integer for chunked MLP")
                specs.append(WeightSpec(name=name, num_blocks=(mlp_expansion * embed_dim) // orth_dim))
                continue
            raise ValueError(f"Unsupported weight name {name}")

        return tuple(specs)

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

        layer_blocks = self.chunk_weights.reshape(
            len(self.blocks), self.blocks_per_layer, self.embed_dim, self.orth_dim
        )

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self._interpolate_pos_encoding(
            x, H // self.patch_embed.patch_size, W // self.patch_embed.patch_size
        )
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk, blocks in zip(self.blocks, layer_blocks):
            x = blk(x, self.reconstruct_layer_weights(blocks))
        x = self.norm(x)
        return x[:, 0]

    def reconstruct_layer_weights(self, blocks: torch.Tensor) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {}
        start = 0
        for spec in self.weight_specs:
            end = start + spec.num_blocks
            matrix = blocks_to_matrix(blocks[start:end]).contiguous()
            if spec.name == "w1":
                weights[spec.name] = matrix.transpose(0, 1).contiguous()
            else:
                weights[spec.name] = matrix
            start = end
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x
