from .vit import VisionTransformer
from .vit_chunk import ChunkedVisionTransformer
from .vit_chunk_atten import ChunkedAttenVisionTransformer
from .vit_chunk_mlp import ChunkedMlpVisionTransformer


def rout_model(**kwargs):
    chunk_type = kwargs.get("chunk_type", "none")
    if "chunk_type" in kwargs:
        kwargs.pop("chunk_type")
    if chunk_type == "none":
        kwargs.pop("orth_dim", None)
        return VisionTransformer(**kwargs)
    elif chunk_type == "all":
        return ChunkedVisionTransformer(**kwargs)
    elif chunk_type == "atten":
        return ChunkedAttenVisionTransformer(**kwargs)
    elif chunk_type == "mlp":
        return ChunkedMlpVisionTransformer(**kwargs)
    else:
        raise ValueError(f"Unsupported chunk_type {chunk_type}!")
