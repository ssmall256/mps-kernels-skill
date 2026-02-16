"""Multi-head RoPE kernel for PyTorch MPS.

Input shape: (batch, heads, seq, dim)
Output shape: (batch, heads, seq, dim)
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.autotune_cache import pick_group_size
from kernels.utils import (
    as_contiguous_2d_lastdim,
    default_group_size,
    kernel_limits,
    require_mps,
    restore_from_2d,
    rowwise_group_candidates,
    validate_rowwise_collective_launch,
)

ROPE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void rope_rowwise(const device float* x,
                         device float* out,
                         constant uint& rows,
                         constant uint& d,
                         constant uint& seq,
                         constant int& offset,
                         uint tid [[thread_index_in_threadgroup]],
                         uint3 tg_pos [[threadgroup_position_in_grid]],
                         uint3 tptg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= rows) {
        return;
    }
    uint tg_size = tptg.x;
    uint half_d = d / 2;
    uint base = row * d;
    uint pos = row % seq;

    for (uint i = tid; i < half_d; i += tg_size) {
        float freq = 1.0f / pow(10000.0f, (2.0f * float(i)) / float(d));
        float angle = float(offset + int(pos)) * freq;
        float c = cos(angle);
        float s = sin(angle);

        float x0 = x[base + i];
        float x1 = x[base + i + half_d];
        out[base + i] = x0 * c - x1 * s;
        out[base + i + half_d] = x1 * c + x0 * s;
    }
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(ROPE_SOURCE)
    return _LIB


def multihead_rope(
    x: torch.Tensor,
    *,
    offset: int = 0,
    group_size: int | None = None,
) -> torch.Tensor:
    require_mps()
    if x.device.type != "mps":
        raise ValueError("x must be on mps")
    if x.ndim != 4:
        raise ValueError("x must be rank-4: (batch, heads, seq, dim)")
    if int(x.shape[-1]) % 2 != 0:
        raise ValueError("dim must be even for RoPE pair rotation")

    seq = int(x.shape[2])
    x2, orig_shape, d = as_contiguous_2d_lastdim(x)
    rows = int(x2.shape[0])

    x32 = x2.to(torch.float32)
    out32 = torch.empty_like(x32)

    kernel = _lib().rope_rowwise
    max_group, simd_width = kernel_limits(kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "multihead_rope: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    def _run_with_group(g: int) -> None:
        validate_rowwise_collective_launch(
            op_name="multihead_rope",
            rows=rows,
            d=d,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel(
            x32,
            out32,
            rows,
            d,
            seq,
            int(offset),
            threads=(rows * int(g),),
            group_size=(int(g),),
        )

    if group_size is None:
        group_size = pick_group_size(
            key=f"rope:rows={rows}:seq={seq}:d={d}:offset={offset}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    _run_with_group(int(group_size))
    y = restore_from_2d(out32, orig_shape)
    return y.to(dtype=x.dtype)


def _reference_multihead_rope(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    x32 = x.to(torch.float32)
    d = int(x32.shape[-1])
    half = d // 2
    seq = int(x32.shape[2])

    pos = torch.arange(seq, device=x.device, dtype=torch.float32) + float(offset)
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = torch.pow(10000.0, -(2.0 * idx / float(d)))
    angles = pos[:, None] * inv_freq[None, :]
    c = torch.cos(angles)[None, None, :, :]
    s = torch.sin(angles)[None, None, :, :]

    x0 = x32[..., :half]
    x1 = x32[..., half:]
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return torch.cat([y0, y1], dim=-1).to(dtype=x.dtype)


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping multihead RoPE validation.")
        return True

    device = torch.device("mps")
    x = torch.randn(2, 8, 128, 128, device=device, dtype=torch.float32)
    y = multihead_rope(x, offset=7)
    ref = _reference_multihead_rope(x, offset=7)
    torch.mps.synchronize()

    max_diff = (y - ref).abs().max().item()
    print(f"multihead_rope max|diff|={max_diff:.3e}")
    return max_diff < 2e-4


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
