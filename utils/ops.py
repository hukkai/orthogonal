import torch


def fast_exp4(x: torch.Tensor) -> torch.Tensor:
    half_x2 = x @ x / 2
    y = x + half_x2
    y = y + x @ half_x2 / 3
    y = y + half_x2 @ half_x2 / 6
    y.diagonal(dim1=-2, dim2=-1).add_(1)
    return y


def fast_exp3(x: torch.Tensor) -> torch.Tensor:
    half_x2 = x @ x / 2
    y = x + half_x2
    y = y + x @ half_x2 / 3
    y.diagonal(dim1=-2, dim2=-1).add_(1)
    return y


def fast_exp2(x: torch.Tensor) -> torch.Tensor:
    half_x2 = x @ x / 2
    y = x + half_x2
    y.diagonal(dim1=-2, dim2=-1).add_(1)
    return y


@torch.no_grad()
def fast_exp(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=(1, 2)).max()
    if norm < 0.05:
        return fast_exp2(x)
    if norm < 0.25:
        return fast_exp3(x)
    if norm < 1:
        return fast_exp4(x)
    return torch.matrix_exp(x)


@torch.no_grad()
def polar(a: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    aa = a.double()
    ata = aa.mT @ aa
    r = (ata - torch.eye(a.shape[-1], device=aa.device, dtype=aa.dtype))
    r = r.norm(dim=(1, 2)).max().item()
    if r < 1e-5:
        return a
    try:
        eigvals, eigvecs = torch.linalg.eigh(ata)
        sqrt_eigvals = eigvals.clamp(min=eps).sqrt().unsqueeze(1)
        ata_inv_sqrt = (eigvecs / sqrt_eigvals) @ eigvecs.mT
        u = aa @ ata_inv_sqrt
        return u.to(a.dtype)
    except torch.linalg.LinAlgError:
        u, _, vt = torch.linalg.svd(aa)
        q = u @ vt
        return q.to(a.dtype)
