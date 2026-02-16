"""Matrix-multiply kernel patterns for PyTorch MPS.

This script provides:
- a basic 2D matmul kernel via `torch.mps.compile_shader`
- guidance point for extension-based `simdgroup_matrix` MMA paths

Note: true `simdgroup_matrix` MMA intrinsics are best handled in extension
pipelines. Use this script as a correctness/dispatch template and compare to
`torch.matmul` for performance decisions.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.utils import kernel_limits, require_mps, validate_threadgroup_size

MATMUL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void matmul_2d(const device float* a,
                      const device float* b,
                      device float* c,
                      constant uint& m,
                      constant uint& k,
                      constant uint& n,
                      uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(MATMUL_SOURCE)
    return _LIB


def matmul_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    group_size: tuple[int, int] = (16, 16),
) -> torch.Tensor:
    require_mps()
    if a.device.type != "mps" or b.device.type != "mps":
        raise ValueError("a and b must be on mps")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2")
    if int(a.shape[1]) != int(b.shape[0]):
        raise ValueError("inner dimensions must match")

    m, k = map(int, a.shape)
    _, n = map(int, b.shape)
    gx, gy = map(int, group_size)
    if gx < 1 or gy < 1:
        raise ValueError("group_size values must be >= 1")

    a32 = a.contiguous().to(torch.float32)
    b32 = b.contiguous().to(torch.float32)
    c32 = torch.empty((m, n), device=a.device, dtype=torch.float32)

    kernel = _lib().matmul_2d
    max_group, _ = kernel_limits(kernel)
    validate_threadgroup_size(
        op_name="matmul_2d",
        group_size=gx * gy,
        max_group=max_group,
    )

    kernel(
        a32,
        b32,
        c32,
        m,
        k,
        n,
        threads=(n, m),
        group_size=(gx, gy),
    )
    return c32.to(dtype=a.dtype)


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping matmul_2d validation.")
        return True

    device = torch.device("mps")
    a = torch.randn(128, 256, device=device, dtype=torch.float32)
    b = torch.randn(256, 64, device=device, dtype=torch.float32)
    y = matmul_2d(a, b)
    ref = torch.matmul(a, b)
    torch.mps.synchronize()

    max_diff = (y - ref).abs().max().item()
    print(f"matmul_2d max|diff|={max_diff:.3e}")
    if max_diff >= 2e-3:
        return False

    print(
        "Note: for simdgroup_matrix MMA paths, use extension patterns "
        "(see references/extension-patterns.md + references/metal-4-spec-notes.md)."
    )
    return True


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
