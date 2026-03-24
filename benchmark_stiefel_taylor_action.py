from __future__ import annotations

import argparse
import time

import torch

from utils.ops import fast_exp, so_proj


def screen_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


@torch.no_grad()
def make_stiefel(batch: int, n: int, m: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(batch, m, n, device=device, dtype=dtype), mode="reduced")
    return q.mT.contiguous()


@torch.no_grad()
def explicit_so_proj(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return so_proj(x, grad)


@torch.no_grad()
def so_proj_fro_norm_formula(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    x_screen = x if x.dtype == screen_dtype(x) else x.to(screen_dtype(x))
    grad_screen = grad if grad.dtype == screen_dtype(grad) else grad.to(screen_dtype(grad))
    gx_t = grad_screen @ x_screen.mT
    grad_norm_sq = (grad_screen * grad_screen).sum(dim=(-2, -1))
    trace_sq = torch.einsum("...ij,...ji->...", gx_t, gx_t)
    return (0.5 * (grad_norm_sq - trace_sq)).clamp_min_(0).sqrt_()


@torch.no_grad()
def taylor_stiefel_action_fused(x: torch.Tensor, grad: torch.Tensor, order: int) -> torch.Tensor:
    """
    Compute X * sum_{k=0}^order A^k / k! for A = asym(X^T G) without forming A.

    Every X A^k stays in span(X, G), so we recurse on n x n coefficients:
        X A^k = C_k X + D_k G.
    """
    if x.shape != grad.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(grad.shape)}")
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")

    *batch_shape, n, _ = x.shape
    s = x @ grad.mT
    m = s.mT
    h = grad @ grad.mT

    eye = torch.eye(n, device=x.device, dtype=x.dtype).expand(*batch_shape, n, n)
    coeff_x = eye.clone()
    coeff_g = torch.zeros_like(s)
    cur_x = eye
    cur_g = torch.zeros_like(s)
    inv_factorial = 1.0

    for k in range(1, order + 1):
        next_x = -0.5 * (cur_x @ s + cur_g @ h)
        next_g = 0.5 * (cur_x + cur_g @ m)
        inv_factorial /= k
        coeff_x = coeff_x + inv_factorial * next_x
        coeff_g = coeff_g + inv_factorial * next_g
        cur_x, cur_g = next_x, next_g

    return coeff_x @ x + coeff_g @ grad


@torch.no_grad()
def taylor_stiefel_action_baseline(x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    a = explicit_so_proj(x, grad)
    return x @ fast_exp(a)


@torch.no_grad()
def taylor_stiefel_action_fused_auto(x: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    norm = so_proj_fro_norm_formula(x, grad).max().item()
    if norm < 0.05:
        return taylor_stiefel_action_fused(x, grad, order=2), 2, norm
    if norm < 0.25:
        return taylor_stiefel_action_fused(x, grad, order=3), 3, norm
    if norm < 1:
        return taylor_stiefel_action_fused(x, grad, order=4), 4, norm
    raise ValueError(
        f"target ||asym(X^T G)||_F = {norm:.4f} reaches the matrix_exp fallback; "
        "this benchmark only targets the Taylor path."
    )


@torch.no_grad()
def scale_grad_to_target_norm(
    x: torch.Tensor,
    grad: torch.Tensor,
    target_norm: float,
    tol: float = 1e-4,
) -> torch.Tensor:
    current = explicit_so_proj(x, grad).norm(dim=(-2, -1)).amax()
    if current.item() == 0:
        raise ValueError("sampled gradient produced zero skew projection")
    grad = grad * (target_norm / current)
    checked = explicit_so_proj(x, grad).norm(dim=(-2, -1)).amax().item()
    if abs(checked - target_norm) > tol * max(1.0, target_norm):
        raise RuntimeError(
            f"failed to match target norm: target={target_norm:.6f}, got={checked:.6f}"
        )
    return grad.contiguous()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark(fn, warmup: int, iters: int, device: str) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return elapsed_ms, out


def parse_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype {name}")
    return table[name]


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run_case(
    batch: int,
    n: int,
    m: int,
    target_norm: float,
    device: str,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    x = make_stiefel(batch, n, m, device=device, dtype=dtype)
    grad = torch.randn(batch, n, m, device=device, dtype=dtype)
    grad = scale_grad_to_target_norm(x, grad, target_norm=target_norm)

    baseline_ms, baseline_out = benchmark(
        lambda: taylor_stiefel_action_baseline(x, grad),
        warmup=warmup,
        iters=iters,
        device=device,
    )

    fused_once, order, actual_norm = taylor_stiefel_action_fused_auto(x, grad)
    fused_ms, fused_out = benchmark(
        lambda: taylor_stiefel_action_fused_auto(x, grad)[0],
        warmup=warmup,
        iters=iters,
        device=device,
    )

    max_abs = (baseline_out - fused_out).abs().max().item()
    rel = (baseline_out - fused_out).norm().item() / baseline_out.norm().item()
    formula_norm = so_proj_fro_norm_formula(x, grad).amax().item()
    fused_check = (fused_once - fused_out).abs().max().item()

    print(
        f"n={n:>3d} m={m:>4d} batch={batch:>3d} "
        f"target={target_norm:>5.2f} actual={actual_norm:>7.4f} formula={formula_norm:>7.4f} "
        f"order={order}"
    )
    print(
        f"  baseline={baseline_ms:>8.3f} ms   fused={fused_ms:>8.3f} ms   "
        f"speedup={baseline_ms / fused_ms:>6.2f}x"
    )
    print(
        f"  max_abs_err={max_abs:.3e}   rel_err={rel:.3e}   fused_repro_err={fused_check:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fused Taylor computation of X exp(asym(X^T G))."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--ns", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--target-norms", type=float, nargs="+", default=[0.03, 0.15, 0.60])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = parse_dtype(args.dtype)

    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise ValueError(f"{dtype} is not a good CPU benchmark dtype here")

    print(f"device={device} dtype={dtype} m={args.m} batch={args.batch}")
    print("benchmark compares current path: X @ fast_exp(so_proj(X, G))")
    print("against a fused Taylor action that never materializes the m x m matrix.")
    print()

    for n in args.ns:
        for target_norm in args.target_norms:
            run_case(
                batch=args.batch,
                n=n,
                m=args.m,
                target_norm=target_norm,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            print()


if __name__ == "__main__":
    main()
