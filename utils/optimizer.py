from __future__ import annotations

from typing import Iterable

import torch


def get_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    exclude_names: Iterable[str] | None = None,
) -> list[dict]:
    exclude_names = set(exclude_names or [])
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(ex in name for ex in exclude_names):
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name
            or "pos_embed" in name
            or "cls_token" in name
            or "gamma_" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
