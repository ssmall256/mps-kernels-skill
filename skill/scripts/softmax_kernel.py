"""Row-wise softmax Metal kernel template for PyTorch MPS."""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Literal, overload

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

SOFTMAX_MULTI_SG_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void softmax_rowwise_multi_sg(const device float* x,
                                     device float* out,
                                     constant uint& rows,
                                     constant uint& d,
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

    float row_max = -INFINITY;
    for (uint i = tid; i < d; i += tg_size) {
        row_max = max(row_max, x[base + i]);
    }
    row_max = simd_max(row_max);

    threadgroup float shared[32];
    if (lane == 0) {
        shared[sg] = row_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float partial = (lane < num_sg) ? shared[lane] : -INFINITY;
        float total = simd_max(partial);
        if (lane == 0) {
            shared[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    row_max = shared[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < d; i += tg_size) {
        float e = exp(x[base + i] - row_max);
        out[base + i] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) {
        shared[sg] = local_sum;
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

    float inv = 1.0f / shared[0];
    for (uint i = tid; i < d; i += tg_size) {
        out[base + i] *= inv;
    }
}
'''

SOFTMAX_SINGLE_SG_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

kernel void softmax_rowwise_single_sg(const device float* x,
                                      device float* out,
                                      constant uint& rows,
                                      constant uint& d,
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint3 tg_pos [[threadgroup_position_in_grid]],
                                      uint3 tptg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= rows) {
        return;
    }

    uint tg_size = tptg.x;
    uint base = row * d;

    float row_max = -INFINITY;
    for (uint i = tid; i < d; i += tg_size) {
        row_max = max(row_max, x[base + i]);
    }
    row_max = simd_max(row_max);

    float local_sum = 0.0f;
    for (uint i = tid; i < d; i += tg_size) {
        float e = exp(x[base + i] - row_max);
        out[base + i] = e;
        local_sum += e;
    }
    float total = simd_sum(local_sum);
    float inv = 1.0f / total;

    for (uint i = tid; i < d; i += tg_size) {
        out[base + i] *= inv;
    }
}
'''

_VARIANT_MULTI = "multi_sg"
_VARIANT_SINGLE = "single_sg"
_LIB_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class SoftmaxLaunchConfig:
    rows: int
    d: int
    group_size: int
    simd_width: int
    max_group: int
    variant: str


def _variant_for_group(group_size: int, simd_width: int) -> str:
    if group_size == simd_width:
        return _VARIANT_SINGLE
    return _VARIANT_MULTI


def _lib_for_variant(variant: str):
    lib = _LIB_CACHE.get(variant)
    if lib is not None:
        return lib

    if variant == _VARIANT_SINGLE:
        source = SOFTMAX_SINGLE_SG_SOURCE
    elif variant == _VARIANT_MULTI:
        source = SOFTMAX_MULTI_SG_SOURCE
    else:
        raise ValueError(f"unknown softmax variant '{variant}'")

    lib = torch.mps.compile_shader(source)
    _LIB_CACHE[variant] = lib
    return lib


def _kernel_for_variant(variant: str):
    if variant == _VARIANT_SINGLE:
        return _lib_for_variant(variant).softmax_rowwise_single_sg
    if variant == _VARIANT_MULTI:
        return _lib_for_variant(variant).softmax_rowwise_multi_sg
    raise ValueError(f"unknown softmax variant '{variant}'")


@overload
def softmax(
    x: torch.Tensor,
    axis: int = -1,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def softmax(
    x: torch.Tensor,
    axis: int = -1,
    group_size: int | None = None,
    *,
    return_launch_config: Literal[True],
) -> tuple[torch.Tensor, SoftmaxLaunchConfig]: ...


def softmax(
    x: torch.Tensor,
    axis: int = -1,
    group_size: int | None = None,
    *,
    return_launch_config: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, SoftmaxLaunchConfig]:
    require_mps()
    if axis not in (-1, x.ndim - 1):
        raise ValueError("custom softmax only supports the last axis")
    if x.device.type != "mps":
        raise ValueError("x must be on mps")

    x2, orig_shape, d = as_contiguous_2d_lastdim(x)
    rows = int(x2.shape[0])

    x32 = x2.to(torch.float32)
    out32 = torch.empty_like(x32)

    probe_kernel = _kernel_for_variant(_VARIANT_MULTI)
    max_group, simd_width = kernel_limits(probe_kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "softmax: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    def _run_with_group(g: int) -> None:
        variant = _variant_for_group(int(g), simd_width)
        if variant == _VARIANT_SINGLE and int(g) != simd_width:
            raise ValueError(
                f"softmax: single_sg variant requires group_size={simd_width}, got {g}"
            )
        validate_rowwise_collective_launch(
            op_name="softmax",
            rows=rows,
            d=d,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel = _kernel_for_variant(variant)
        kernel(
            x32,
            out32,
            rows,
            d,
            threads=(rows * int(g),),
            group_size=(int(g),),
        )

    if group_size is None:
        group_size = pick_group_size(
            key=f"softmax:rows={rows}:d={d}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    chosen_group = int(group_size)
    _run_with_group(chosen_group)
    variant = _variant_for_group(chosen_group, simd_width)

    y = restore_from_2d(out32, orig_shape)
    y = y.to(dtype=x.dtype)
    if not return_launch_config:
        return y

    return y, SoftmaxLaunchConfig(
        rows=rows,
        d=d,
        group_size=chosen_group,
        simd_width=simd_width,
        max_group=max_group,
        variant=variant,
    )


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping softmax validation.")
        return True

    device = torch.device("mps")
    shapes = [(2, 128), (4, 64, 256), (1, 8, 1024), (2, 2, 32)]
    ok = True

    for shape in shapes:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        y = softmax(x)
        ref = torch.softmax(x, dim=-1)
        torch.mps.synchronize()

        max_diff = (y - ref).abs().max().item()
        sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
        print(f"shape={shape} max|diff|={max_diff:.3e} sum_err={sum_err:.3e}")
        ok = ok and max_diff < 1e-5 and sum_err < 1e-5

    return ok


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
