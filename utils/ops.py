import torch


@torch.no_grad()
def polar(a: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    if a.ndim < 2:
        raise ValueError("polar expects a matrix or batch of matrices")

    aa = a.double()
    ata = aa.mT @ aa
    r = ata - torch.eye(a.shape[-1], device=aa.device, dtype=aa.dtype)
    r = r.norm(dim=(-2, -1)).amax().item()
    if r < 1e-5:
        return a
    try:
        eigvals, eigvecs = torch.linalg.eigh(ata)
        sqrt_eigvals = eigvals.clamp(min=eps).sqrt().unsqueeze(-2)
        ata_inv_sqrt = (eigvecs / sqrt_eigvals) @ eigvecs.mT
        u = aa @ ata_inv_sqrt
        return u.to(a.dtype)
    except Exception:
        u, _, vt = torch.linalg.svd(aa, full_matrices=False)
        q = u @ vt
        return q.to(a.dtype)
