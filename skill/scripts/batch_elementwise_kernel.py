"""Batched SiLU example using torch.mps.compile_shader."""
from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.autotune_cache import pick_group_size
from kernels.utils import kernel_limits, require_mps, validate_threadgroup_size

SILU_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void silu_kernel(const device float* x,
                        device float* out,
                        constant uint& n,
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) {
        return;
    }
    float v = x[i];
    out[i] = v / (1.0f + exp(-v));
}
'''

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(SILU_SOURCE)
    return _LIB


def batch_silu(x: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    require_mps()
    if x.device.type != "mps":
        raise ValueError("x must be on mps device")

    x32 = x.contiguous().to(torch.float32)
    out = torch.empty_like(x32)
    n = int(x32.numel())
    kernel = _lib().silu_kernel

    max_group, _ = kernel_limits(kernel)
    default_group = max(32, min(256, max_group))

    if group_size is None:
        candidates = tuple(g for g in (32, 64, 128, 256, 512) if g <= max_group)
        if not candidates:
            candidates = (max_group,)

        def _run_with_group(g: int) -> None:
            validate_threadgroup_size(
                op_name="batch_silu",
                group_size=int(g),
                max_group=max_group,
            )
            kernel(x32, out, n, threads=(n,), group_size=(int(g),))

        group_size = pick_group_size(
            key=f"silu:n={n}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    validate_threadgroup_size(
        op_name="batch_silu",
        group_size=int(group_size),
        max_group=max_group,
    )
    kernel(x32, out, n, threads=(n,), group_size=(int(group_size),))
    return out.to(dtype=x.dtype)


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping SiLU validation.")
        return True

    device = torch.device("mps")
    x = torch.randn(4, 256, 1024, device=device, dtype=torch.float32)
    y = batch_silu(x)
    torch.mps.synchronize()

    ref = torch.nn.functional.silu(x)
    max_diff = (y - ref).abs().max().item()
    print(f"batch_silu max|diff|={max_diff:.3e}")
    return max_diff < 1e-5


if __name__ == "__main__":
    ok = validate()
    print("PASS" if ok else "FAIL")
