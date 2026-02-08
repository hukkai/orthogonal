from __future__ import annotations

import torch
import torch.distributed as dist

from .ops import fast_exp, polar, so_proj


class SOOptimizer:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        project_last: bool = True,
    ) -> None:
        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.project_last = project_last

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        total = param.shape[0]
        if total % self.world_size != 0:
            raise ValueError("chunk_weights must be divisible by world size")

        per_rank = total // self.world_size
        self.local_slice = slice(self.rank * per_rank, (self.rank + 1) * per_rank)

        self.m = torch.zeros_like(param.data[self.local_slice])
        self.v = torch.ones_like(self.m) / (param.shape[-1] ** 2)
        self.buffer = torch.zeros_like(param.data)

    def state_dict(self) -> dict:
        return {
            "m": self.m,
            "v": self.v,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "project_last": self.project_last,
        }

    def load_state_dict(self, state: dict) -> None:
        self.m = state.get("m", self.m)
        self.v = state.get("v", self.v)
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.project_last = state.get("project_last", self.project_last)

    def step(self, lr: float | None = None, is_last: bool = False) -> None:
        if self.param.grad is None:
            return
        lr = lr if lr is not None else self.lr

        x = self.param.data[self.local_slice]
        grad = self.param.grad[self.local_slice]

        grad = so_proj(x, grad)
        grad = grad / grad.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)

        self.m += (grad - self.m) * (1.0 - self.beta1)
        self.v += (grad.pow(2) - self.v) * (1.0 - self.beta2)

        update = -self.m / (self.v.sqrt() + self.eps) * lr
        update = fast_exp(update.double())
        new_x = (x.double() @ update).to(x.dtype)

        if is_last and self.project_last:
            new_x = polar(new_x)

        self.buffer.zero_()
        self.buffer[self.local_slice] = new_x
        if dist.is_initialized():
            dist.all_reduce(self.buffer)
        self.param.data.copy_(self.buffer)
        self.param.grad = None

    def finish_epoch(self) -> None:
        self.m = 0.5 * (self.m - self.m.transpose(-1, -2))
        self.v = 0.5 * (self.v + self.v.transpose(-1, -2))
