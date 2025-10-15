"""Tensorized LLAMA-style language model implemented with PyTorch.

This module demonstrates how to organize all repeated layer weights as
model-level tensors instead of instantiating per-layer ``nn.Module``
blocks.  Every group of weights that is typically duplicated per layer
is represented as a single ``nn.Parameter`` whose first dimension is the
layer index.

The implementation intentionally mirrors the high-level structure of the
Llama 3 architecture (rotary attention, SiLU-gated feed-forward network,
RMSNorm, tied embeddings) while remaining compact enough to serve as a
standalone reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding as _TransformersLlamaRotaryEmbedding,
        rms_norm as _transformers_rms_norm,
    )
except Exception:  # pragma: no cover - transformers is optional
    _TransformersLlamaRotaryEmbedding = None
    _transformers_rms_norm = None


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm using ``transformers`` when available."""

    if _transformers_rms_norm is not None:
        return _transformers_rms_norm(hidden_states, weight, eps)

    norm = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(norm + eps)
    return hidden_states * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings with a ``transformers`` fallback."""

    def __init__(self, head_dim: int, max_position_embeddings: int, base: int = 10000):
        super().__init__()

        self.head_dim = head_dim
        if _TransformersLlamaRotaryEmbedding is not None:
            self._impl: Optional[nn.Module] = _TransformersLlamaRotaryEmbedding(
                dim=head_dim, max_position_embeddings=max_position_embeddings, base=base
            )
        else:
            self._impl = None
            inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("_cached_cos", torch.empty(0), persistent=False)
            self.register_buffer("_cached_sin", torch.empty(0), persistent=False)
            self._seq_len_cached = 0

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._impl is not None:
            dummy = torch.zeros(1, seq_len, 1, self.head_dim, device=device, dtype=dtype)
            try:
                cos, sin = self._impl(dummy, seq_len=seq_len)
            except TypeError:
                cos, sin = self._impl(dummy)
            cos = cos.to(device=device, dtype=dtype)
            sin = sin.to(device=device, dtype=dtype)
            cos = cos.reshape(-1, self.head_dim)[:seq_len]
            sin = sin.reshape(-1, self.head_dim)[:seq_len]
            return cos, sin

        assert hasattr(self, "inv_freq"), "fallback buffers not initialized"
        if (
            seq_len <= self._seq_len_cached
            and self._cached_cos.device == device
            and self._cached_cos.dtype == dtype
        ):
            cos = self._cached_cos[:seq_len]
            sin = self._cached_sin[:seq_len]
            return cos, sin

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        self._cached_cos = cos
        self._cached_sin = sin
        self._seq_len_cached = seq_len
        return cos, sin


@dataclass
class TensorizedLlamaConfig:
    """Configuration for :class:`TensorizedLlamaModel`."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: int = 10000
    init_std: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.hidden_size % self.num_key_value_heads != 0:
            raise ValueError("hidden_size must be divisible by num_key_value_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class TensorizedLlamaModel(nn.Module):
    """Llama-style decoder-only transformer with tensorized layer weights."""

    def __init__(self, config: TensorizedLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_head_dim = self.head_dim
        self.kv_dim = self.kv_head_dim * config.num_key_value_heads

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Layer parameters grouped by tensor.
        L = config.num_hidden_layers
        H = config.hidden_size
        I = config.intermediate_size
        self.input_layernorm_weight = nn.Parameter(torch.ones(L, H))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(L, H))

        self.q_proj_weight = nn.Parameter(torch.empty(L, H, H))
        self.kv_proj_weight = nn.Parameter(torch.empty(L, H, 2 * self.kv_dim))
        self.o_proj_weight = nn.Parameter(torch.empty(L, H, H))

        self.ffn_gate_up_weight = nn.Parameter(torch.empty(L, H, 2 * I))
        self.ffn_down_weight = nn.Parameter(torch.empty(L, I, H))

        self.final_layernorm_weight = nn.Parameter(torch.ones(H))

        self.lm_head = nn.Linear(H, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.init_std)
        nn.init.normal_(self.q_proj_weight, mean=0.0, std=self.config.init_std)
        nn.init.normal_(self.kv_proj_weight, mean=0.0, std=self.config.init_std)
        nn.init.normal_(self.o_proj_weight, mean=0.0, std=self.config.init_std)
        nn.init.normal_(self.ffn_gate_up_weight, mean=0.0, std=self.config.init_std)
        nn.init.normal_(self.ffn_down_weight, mean=0.0, std=self.config.init_std)
        nn.init.ones_(self.input_layernorm_weight)
        nn.init.ones_(self.post_attention_layernorm_weight)
        nn.init.ones_(self.final_layernorm_weight)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.init_std)

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def _expand_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, dtype: torch.dtype
    ) -> torch.Tensor:
        # Convert 1 -> 0, 0 -> large negative for masked positions.
        expanded_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(dtype).min
        return expanded_mask.expand(-1, 1, target_length, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)

        causal_mask = self._make_causal_mask(seq_len, device, hidden_states.dtype)
        if attention_mask is not None:
            attn_mask = self._expand_attention_mask(attention_mask, seq_len, hidden_states.dtype)
        else:
            attn_mask = None

        cos, sin = self.rotary_emb(seq_len, device=device, dtype=hidden_states.dtype)

        for layer_idx in range(self.config.num_hidden_layers):
            residual = hidden_states

            normed_states = _rms_norm(
                hidden_states,
                self.input_layernorm_weight[layer_idx],
                self.config.rms_norm_eps,
            )

            q = torch.matmul(normed_states, self.q_proj_weight[layer_idx])
            kv = torch.matmul(normed_states, self.kv_proj_weight[layer_idx])
            k, v = torch.chunk(kv, 2, dim=-1)

            q = q.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.config.num_key_value_heads, self.kv_head_dim)
            v = v.view(batch_size, seq_len, self.config.num_key_value_heads, self.kv_head_dim)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

            if self.config.num_key_value_heads != self.config.num_attention_heads:
                repeat_factor = self.config.num_attention_heads // self.config.num_key_value_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_scores = attn_scores + causal_mask
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            attn_probs = F.softmax(attn_scores, dim=-1, dtype=hidden_states.dtype)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            attn_output = torch.matmul(attn_output, self.o_proj_weight[layer_idx])

            hidden_states = residual + attn_output

            residual = hidden_states
            normed_states = _rms_norm(
                hidden_states,
                self.post_attention_layernorm_weight[layer_idx],
                self.config.rms_norm_eps,
            )

            gate_up = torch.matmul(normed_states, self.ffn_gate_up_weight[layer_idx])
            gate, up = gate_up.split(self.config.intermediate_size, dim=-1)
            gate = F.silu(gate)
            hidden_states = residual + torch.matmul(gate * up, self.ffn_down_weight[layer_idx])

        hidden_states = _rms_norm(hidden_states, self.final_layernorm_weight, self.config.rms_norm_eps)

        if return_logits:
            return self.lm_head(hidden_states)
        return hidden_states


def demo() -> None:
    """Run a quick forward pass to validate the implementation."""

    config = TensorizedLlamaConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
    )

    model = TensorizedLlamaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)


if __name__ == "__main__":
    demo()

