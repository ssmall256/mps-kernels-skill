"""LayerNorm Metal kernel template for PyTorch MPS."""
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

LAYERNORM_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void layernorm_rowwise(const device float* x,
                              const device float* w,
                              const device float* b,
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
    uint base = row * d;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (uint i = tid; i < d; i += tg_size) {
        float v = x[base + i];
        local_sum += v;
        local_sq += v * v;
    }
    local_sum = simd_sum(local_sum);
    local_sq = simd_sum(local_sq);

    threadgroup float shared_sum[32];
    threadgroup float shared_sq[32];
    if (lane == 0) {
        shared_sum[sg] = local_sum;
        shared_sq[sg] = local_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float partial_sum = (lane < num_sg) ? shared_sum[lane] : 0.0f;
        float partial_sq = (lane < num_sg) ? shared_sq[lane] : 0.0f;
        float total_sum = simd_sum(partial_sum);
        float total_sq = simd_sum(partial_sq);
        if (lane == 0) {
            shared_sum[0] = total_sum;
            shared_sq[0] = total_sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = shared_sum[0] / float(d);
    float var = max(shared_sq[0] / float(d) - mean * mean, 0.0f);
    float inv_std = rsqrt(var + eps);

    for (uint i = tid; i < d; i += tg_size) {
        float n = (x[base + i] - mean) * inv_std;
        out[base + i] = n * w[i] + b[i];
    }
}
'''

_LIB = None


@dataclass(frozen=True)
class LayerNormLaunchConfig:
    rows: int
    d: int
    group_size: int
    simd_width: int
    max_group: int


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(LAYERNORM_SOURCE)
    return _LIB


@overload
def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[True],
) -> tuple[torch.Tensor, LayerNormLaunchConfig]: ...


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    group_size: int | None = None,
    *,
    return_launch_config: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LayerNormLaunchConfig]:
    require_mps()
    if x.device.type != "mps" or weight.device.type != "mps" or bias.device.type != "mps":
        raise ValueError("x, weight, and bias must be on mps")
    if weight.shape != bias.shape:
        raise ValueError("weight and bias must have same shape")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight shape must match x.shape[-1]")

    x2, orig_shape, d = as_contiguous_2d_lastdim(x)
    rows = int(x2.shape[0])

    x32 = x2.to(torch.float32)
    w32 = weight.contiguous().to(torch.float32)
    b32 = bias.contiguous().to(torch.float32)
    out32 = torch.empty_like(x32)

    kernel = _lib().layernorm_rowwise
    max_group, simd_width = kernel_limits(kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "layernorm: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    def _run_with_group(g: int) -> None:
        validate_rowwise_collective_launch(
            op_name="layernorm",
            rows=rows,
            d=d,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel(
            x32,
            w32,
            b32,
            out32,
            rows,
            d,
            float(eps),
            threads=(rows * int(g),),
            group_size=(int(g),),
        )

    if group_size is None:
        group_size = pick_group_size(
            key=f"layernorm:rows={rows}:d={d}",
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

    return y, LayerNormLaunchConfig(
        rows=rows,
        d=d,
        group_size=chosen_group,
        simd_width=simd_width,
        max_group=max_group,
    )


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping LayerNorm validation.")
        return True

    device = torch.device("mps")
    shapes = [(2, 128), (4, 64, 256), (1, 8, 1024)]
    ok = True

    for shape in shapes:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32)
        b = torch.randn(shape[-1], device=device, dtype=torch.float32)

        y = layernorm(x, w, b, eps=1e-5)
        ref = F.layer_norm(x, (shape[-1],), w, b, eps=1e-5)
        torch.mps.synchronize()

        max_diff = (y - ref).abs().max().item()
        print(f"shape={shape} max|diff|={max_diff:.3e}")
        ok = ok and max_diff < 1e-5

    return ok


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
