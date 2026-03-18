from __future__ import annotations

import torch
import torch.distributed as dist

from .ops import polar


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _stiefel_project(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return grad - x @ _symmetrize(x.transpose(-1, -2) @ grad)


def _local_block_slice(total: int, world_size: int, rank: int) -> slice:
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    length = base + int(rank < remainder)
    return slice(start, start + length)


class BlockStiefelAdam:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        project_last: bool | None = None,
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

        self.local_slice = _local_block_slice(param.shape[0], self.world_size, self.rank)
        self.m = torch.zeros_like(param.data[self.local_slice])
        self.v = torch.zeros_like(self.m)
        self.buffer = torch.zeros_like(param.data)
        self.step_count = 0

    @property
    def local_count(self) -> int:
        return self.local_slice.stop - self.local_slice.start

    @torch.no_grad()
    def step(self, lr: float | None = None, is_last: bool = False) -> None:
        del is_last
        if self.param.grad is None:
            return

        lr = lr if lr is not None else self.lr
        self.step_count += 1
        self.buffer.zero_()

        if self.local_count:
            x = self.param.data[self.local_slice]
            grad = self.param.grad[self.local_slice]

            riemannian_grad = _stiefel_project(x, grad)
            self.m.mul_(self.beta1).add_(riemannian_grad, alpha=1.0 - self.beta1)
            self.v.mul_(self.beta2).addcmul_(riemannian_grad, riemannian_grad, value=1.0 - self.beta2)

            m_hat = self.m / (1.0 - self.beta1**self.step_count)
            v_hat = self.v / (1.0 - self.beta2**self.step_count)
            delta_tilde = -lr * m_hat / (v_hat.sqrt() + self.eps)
            delta = _stiefel_project(x, delta_tilde)
            new_x = polar((x + delta).double()).to(x.dtype)
            self.buffer[self.local_slice].copy_(new_x)

        if dist.is_initialized():
            dist.all_reduce(self.buffer)

        self.param.data.copy_(self.buffer)
        self.param.grad = None
