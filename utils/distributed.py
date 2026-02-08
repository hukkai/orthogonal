import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    return False, 0, 0, 1


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor
