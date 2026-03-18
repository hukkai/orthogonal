import unittest

import torch

from models import (
    ChunkedAttenVisionTransformer,
    ChunkedMlpVisionTransformer,
    ChunkedVisionTransformer,
    VisionTransformer,
)
from utils import BlockStiefelAdam
from utils.ops import polar


def _orthonormal_blocks(num_blocks: int, dim: int, orth_dim: int) -> torch.Tensor:
    return torch.linalg.qr(
        torch.randn(num_blocks, dim, orth_dim, dtype=torch.float64),
        mode="reduced",
    )[0].float()


class BlockStiefelRefactorTests(unittest.TestCase):
    def test_invalid_orth_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            ChunkedVisionTransformer(embed_dim=64, depth=1, num_heads=8, mlp_ratio=4.0, orth_dim=10)

    def test_chunk_weight_shapes_all_modes(self) -> None:
        cases = [
            (ChunkedVisionTransformer, "all"),
            (ChunkedAttenVisionTransformer, "atten"),
            (ChunkedMlpVisionTransformer, "mlp"),
        ]
        depth = 2
        embed_dim = 64
        mlp_ratio = 4

        for orth_dim in (64, 16, 1):
            attn_blocks = embed_dim // orth_dim
            ffn_blocks = (mlp_ratio * embed_dim) // orth_dim
            expected_per_layer = {
                "all": 4 * attn_blocks + 2 * ffn_blocks,
                "atten": 4 * attn_blocks,
                "mlp": 2 * ffn_blocks,
            }

            for cls, mode in cases:
                model = cls(
                    img_size=32,
                    patch_size=8,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=8,
                    mlp_ratio=float(mlp_ratio),
                    num_classes=10,
                    orth_dim=orth_dim,
                )
                self.assertEqual(
                    tuple(model.chunk_weights.shape),
                    (depth * expected_per_layer[mode], embed_dim, orth_dim),
                )

    def test_dense_reconstruction_shapes(self) -> None:
        model = ChunkedVisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=64,
            depth=1,
            num_heads=8,
            mlp_ratio=4.0,
            num_classes=10,
            orth_dim=16,
        )
        weights = model.reconstruct_layer_weights(model.chunk_weights[: model.blocks_per_layer])

        self.assertEqual(tuple(weights["q"].shape), (64, 64))
        self.assertEqual(tuple(weights["k"].shape), (64, 64))
        self.assertEqual(tuple(weights["v"].shape), (64, 64))
        self.assertEqual(tuple(weights["proj"].shape), (64, 64))
        self.assertEqual(tuple(weights["w1"].shape), (256, 64))
        self.assertEqual(tuple(weights["w2"].shape), (64, 256))

    def test_optimizer_preserves_block_orthogonality(self) -> None:
        param = torch.nn.Parameter(_orthonormal_blocks(5, 8, 3))
        optimizer = BlockStiefelAdam(param, lr=1e-2)

        for _ in range(5):
            param.grad = torch.randn_like(param)
            optimizer.step()

        gram = param.transpose(-1, -2) @ param
        eye = torch.eye(3, dtype=gram.dtype).expand_as(gram)
        self.assertLess((gram - eye).abs().max().item(), 1e-5)

    def test_polar_supports_2d_and_batched_rectangular_inputs(self) -> None:
        for shape in ((8, 3), (5, 8, 3), (4, 16, 1)):
            projected = polar(torch.randn(*shape))
            gram = projected.transpose(-1, -2) @ projected
            eye = torch.eye(projected.shape[-1], dtype=projected.dtype).expand_as(gram)
            self.assertLess((gram - eye).abs().max().item(), 1e-5)

    def test_forward_smoke(self) -> None:
        x = torch.randn(2, 3, 32, 32)
        models = [
            VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                mlp_ratio=4.0,
                num_classes=10,
            ),
            ChunkedVisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                mlp_ratio=4.0,
                num_classes=10,
                orth_dim=16,
            ),
            ChunkedVisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                mlp_ratio=4.0,
                num_classes=10,
                orth_dim=1,
            ),
            ChunkedAttenVisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                mlp_ratio=4.0,
                num_classes=10,
                orth_dim=16,
            ),
            ChunkedMlpVisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=32,
                depth=1,
                num_heads=4,
                mlp_ratio=4.0,
                num_classes=10,
                orth_dim=1,
            ),
        ]

        for model in models:
            output = model(x)
            self.assertEqual(tuple(output.shape), (2, 10))

if __name__ == "__main__":
    unittest.main()
