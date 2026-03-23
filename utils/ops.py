import torch


def fast_exp4(x: torch.Tensor) -> torch.Tensor:
    x2 = x @ x
    t = x2.mul(1.0 / 24.0).add_(x, alpha=1.0 / 6.0)
    t.diagonal(dim1=-2, dim2=-1).add_(0.5)
    y = x2 @ t
    y.add_(x)
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
def polar(
    a: torch.Tensor,
    tolerance: float = 1e-5,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Project a batch of matrices A in R^{b x n x m} (n <= m)
    onto the row-orthonormal Stiefel manifold:
        Q Q^T = I_n

    Uses:
        Q = (A A^T)^(-1/2) A
    with SVD fallback.

    Args:
        a: Tensor of shape (b, n, m), with n <= m
        tolerance: only project matrices whose ||A A^T - I||_F exceeds this threshold
        eps: eigenvalue clamp for numerical stability

    Returns:
        Tensor of same shape as a, with projected matrices.
    """
    if a.ndim != 3:
        raise ValueError(f"expected a to have shape (b, n, m), got {tuple(a.shape)}")

    b, n, m = a.shape
    if n > m:
        raise ValueError(f"expected n <= m, got shape {tuple(a.shape)}")

    screen_dtype = (
        torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    )
    a_screen = a if a.dtype == screen_dtype else a.to(screen_dtype)

    I = torch.eye(n, device=a.device, dtype=screen_dtype)
    aat = a_screen @ a_screen.transpose(-1, -2)
    err = torch.linalg.matrix_norm(aat - I, ord="fro", dim=(-2, -1))
    mask = err > tolerance

    if not mask.any():
        return a

    a_bad = a[mask].to(torch.float64)
    aat_bad = a_bad @ a_bad.transpose(-1, -2)

    try:
        eigvals, eigvecs = torch.linalg.eigh(aat_bad)
        inv_sqrt = eigvals.clamp_min(eps).rsqrt()
        aat_inv_sqrt = (eigvecs * inv_sqrt.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        q_bad = aat_inv_sqrt @ a_bad
    except RuntimeError:
        # For A in R^{n x m}, n <= m:
        # A = U S V^T  => nearest row-orthonormal Q = U V^T
        u, _, vh = torch.linalg.svd(a_bad, full_matrices=False)
        q_bad = u @ vh

    out = a.clone()
    out[mask] = q_bad.to(a.dtype)

    return out


@torch.no_grad()
def so_proj(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    proj_grad = x.mT @ grad
    proj_grad = 0.5 * (proj_grad - proj_grad.mT)
    return proj_grad
