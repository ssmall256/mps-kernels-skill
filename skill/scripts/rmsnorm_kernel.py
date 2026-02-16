"""RMSNorm Metal kernel template for PyTorch MPS.

Uses row-parallel threadgroup reductions with float32 accumulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Literal, overload

import torch
import torch.nn.functional as F

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

RMSNORM_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_rowwise(const device float* x,
                            const device float* w,
                            device float* out,
                            constant uint& rows,
                            constant uint& d,
                            constant float& eps,
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint3 tg_pos [[threadgroup_position_in_grid]],
                            uint3 tptg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= rows) {
        return;
    }

    uint tg_size = tptg.x;
    uint num_sg = (tg_size + 31) / 32;
    float sum_sq = 0.0f;
    uint base = row * d;
    for (uint i = tid; i < d; i += tg_size) {
        float v = x[base + i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    threadgroup float shared[32];
    if (lane == 0) {
        shared[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float partial = (lane < num_sg) ? shared[lane] : 0.0f;
        float total = simd_sum(partial);
        if (lane == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = rsqrt(shared[0] / float(d) + eps);

    for (uint i = tid; i < d; i += tg_size) {
        out[base + i] = x[base + i] * inv_rms * w[i];
    }
}
'''

_LIB = None


@dataclass(frozen=True)
class RmsNormLaunchConfig:
    rows: int
    d: int
    group_size: int
    simd_width: int
    max_group: int


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(RMSNORM_SOURCE)
    return _LIB


@overload
def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[True],
) -> tuple[torch.Tensor, RmsNormLaunchConfig]: ...


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, RmsNormLaunchConfig]:
    require_mps()
    if x.device.type != "mps":
        raise ValueError("x must be on mps")
    if weight.device.type != "mps":
        raise ValueError("weight must be on mps")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight shape must match x.shape[-1]")

    x2, orig_shape, d = as_contiguous_2d_lastdim(x)
    rows = int(x2.shape[0])
    x32 = x2.to(torch.float32)
    w32 = weight.contiguous().to(torch.float32)
    out32 = torch.empty_like(x32)

    kernel = _lib().rmsnorm_rowwise
    max_group, simd_width = kernel_limits(kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "rmsnorm: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    def _run_with_group(g: int) -> None:
        validate_rowwise_collective_launch(
            op_name="rmsnorm",
            rows=rows,
            d=d,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel(
            x32,
            w32,
            out32,
            rows,
            d,
            float(eps),
            threads=(rows * int(g),),
            group_size=(int(g),),
        )

    if group_size is None:
        group_size = pick_group_size(
            key=f"rmsnorm:rows={rows}:d={d}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    chosen_group = int(group_size)
    _run_with_group(chosen_group)

    y = restore_from_2d(out32, orig_shape)
    y = y.to(dtype=x.dtype)
    if not return_launch_config:
        return y

    return y, RmsNormLaunchConfig(
        rows=rows,
        d=d,
        group_size=chosen_group,
        simd_width=simd_width,
        max_group=max_group,
    )


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping RMSNorm validation.")
        return True

    device = torch.device("mps")
    shapes = [(2, 128), (4, 64, 256), (1, 8, 1024), (2, 2, 32)]
    ok = True

    for shape in shapes:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32)

        y = rmsnorm(x, w, eps=1e-5)
        ref = F.rms_norm(x, (shape[-1],), w, eps=1e-5)
        torch.mps.synchronize()

        max_diff = (y - ref).abs().max().item()
        print(f"shape={shape} max|diff|={max_diff:.3e}")
        ok = ok and max_diff < 1e-5

    return ok


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
