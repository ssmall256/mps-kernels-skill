"""4-bit dequantized matvec kernel for PyTorch MPS.

Demonstrates:
- packed 4-bit weight format (`int32`, 8 values per word)
- fused dequantization + matrix-vector multiply in one kernel
- group-wise scale/bias dequant parameters
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.autotune_cache import pick_group_size
from kernels.utils import (
    default_group_size,
    kernel_limits,
    require_mps,
    rowwise_group_candidates,
    validate_rowwise_collective_launch,
)

DEQUANT_MATVEC_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void dequant_matvec(const device int* w_packed,
                           const device float* scales,
                           const device float* biases,
                           const device float* x,
                           device float* out,
                           constant uint& out_features,
                           constant uint& in_features,
                           constant uint& group_size,
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint3 tg_pos [[threadgroup_position_in_grid]],
                           uint3 tptg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    if (row >= out_features) {
        return;
    }

    uint tg_size = tptg.x;
    uint packed_per_row = in_features / 8;
    uint groups_per_row = in_features / group_size;

    float acc = 0.0f;
    for (uint i = tid; i < in_features; i += tg_size) {
        uint packed_idx = row * packed_per_row + (i / 8);
        uint nibble_idx = i % 8;
        uint packed = as_type<uint>(w_packed[packed_idx]);
        uint nibble = (packed >> (nibble_idx * 4)) & 0xFu;

        uint group_idx = row * groups_per_row + (i / group_size);
        float scale = scales[group_idx];
        float bias = biases[group_idx];
        float w = scale * float(nibble) + bias;
        acc += w * x[i];
    }

    acc = simd_sum(acc);
    threadgroup float shared[32];
    if (lane == 0) {
        shared[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_sg = (tg_size + 31) / 32;
    if (sg == 0) {
        float partial = (lane < num_sg) ? shared[lane] : 0.0f;
        float total = simd_sum(partial);
        if (lane == 0) {
            out[row] = total;
        }
    }
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(DEQUANT_MATVEC_SOURCE)
    return _LIB


def quantize_4bit(
    weight: torch.Tensor,
    *,
    group_size: int = 64,
    output_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("weight must be rank-2: (out_features, in_features)")
    if group_size <= 0 or (group_size % 8) != 0:
        raise ValueError("group_size must be > 0 and divisible by 8")

    w = weight.detach().to(torch.float32).cpu().contiguous()
    out_features, in_features = map(int, w.shape)
    if (in_features % group_size) != 0:
        raise ValueError("in_features must be divisible by group_size")

    groups = in_features // group_size
    wg = w.view(out_features, groups, group_size)

    w_min = wg.amin(dim=-1, keepdim=True)
    w_max = wg.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min) / 15.0
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    bias = w_min

    q = torch.round((wg - bias) / scale).clamp(0.0, 15.0).to(torch.int32)
    q = q.view(out_features, groups, group_size // 8, 8)
    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 1, 8) * 4)
    packed = torch.bitwise_left_shift(q, shifts).sum(dim=-1).to(torch.int32)

    packed = packed.view(out_features, in_features // 8).contiguous()
    scales = scale.squeeze(-1).contiguous().to(torch.float32)
    biases = bias.squeeze(-1).contiguous().to(torch.float32)

    if output_device is not None:
        packed = packed.to(output_device)
        scales = scales.to(output_device)
        biases = biases.to(output_device)
    return packed, scales, biases


def dequantize_4bit(
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    in_features: int,
    group_size: int,
) -> torch.Tensor:
    if w_packed.ndim != 2:
        raise ValueError("w_packed must be rank-2: (out_features, in_features // 8)")
    if scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("scales and biases must be rank-2: (out_features, groups)")

    wp = w_packed.to(torch.int32).cpu().contiguous()
    sc = scales.to(torch.float32).cpu().contiguous()
    bs = biases.to(torch.float32).cpu().contiguous()

    out_features = int(wp.shape[0])
    if int(wp.shape[1]) * 8 != int(in_features):
        raise ValueError("packed width does not match in_features")
    groups = int(in_features) // int(group_size)
    if sc.shape != (out_features, groups) or bs.shape != (out_features, groups):
        raise ValueError("scales/biases shape mismatch")

    shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
    nibbles = torch.bitwise_and(torch.bitwise_right_shift(wp.unsqueeze(-1), shifts), 0xF)
    nibbles = nibbles.to(torch.float32).view(out_features, in_features)
    nibbles = nibbles.view(out_features, groups, group_size)
    return (nibbles * sc.unsqueeze(-1) + bs.unsqueeze(-1)).view(out_features, in_features)


def dequant_matvec(
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    x: torch.Tensor,
    *,
    group_size: int = 64,
    group_threads: int | None = None,
) -> torch.Tensor:
    require_mps()
    if any(t.device.type != "mps" for t in (w_packed, scales, biases, x)):
        raise ValueError("w_packed, scales, biases, x must all be on mps")
    if w_packed.ndim != 2:
        raise ValueError("w_packed must be rank-2")
    if scales.ndim != 2 or biases.ndim != 2:
        raise ValueError("scales/biases must be rank-2")
    if x.ndim != 1:
        raise ValueError("x must be rank-1")
    if scales.shape != biases.shape:
        raise ValueError("scales and biases must have same shape")
    if group_size <= 0 or (group_size % 8) != 0:
        raise ValueError("group_size must be > 0 and divisible by 8")

    out_features = int(w_packed.shape[0])
    in_features = int(x.numel())
    if int(w_packed.shape[1]) * 8 != in_features:
        raise ValueError("w_packed shape does not match x length")
    if (in_features % group_size) != 0:
        raise ValueError("x length must be divisible by group_size")
    groups = in_features // group_size
    if scales.shape != (out_features, groups):
        raise ValueError("scales/biases shape must be (out_features, in_features // group_size)")

    wp = w_packed.contiguous().to(torch.int32)
    sc = scales.contiguous().to(torch.float32)
    bs = biases.contiguous().to(torch.float32)
    xv = x.contiguous().to(torch.float32)
    out = torch.empty(out_features, device=x.device, dtype=torch.float32)

    kernel = _lib().dequant_matvec
    max_group, simd_width = kernel_limits(kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "dequant_matvec: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    def _run_with_group(g: int) -> None:
        validate_rowwise_collective_launch(
            op_name="dequant_matvec",
            rows=out_features,
            d=in_features,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel(
            wp,
            sc,
            bs,
            xv,
            out,
            out_features,
            in_features,
            int(group_size),
            threads=(out_features * int(g),),
            group_size=(int(g),),
        )

    if group_threads is None:
        group_threads = pick_group_size(
            key=f"dequant_matvec:o={out_features}:i={in_features}:g={group_size}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    _run_with_group(int(group_threads))
    return out.to(dtype=x.dtype)


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping dequant matvec validation.")
        return True

    device = torch.device("mps")
    out_features, in_features, group_size = 256, 512, 64
    w = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    x = torch.randn(in_features, device=device, dtype=torch.float32)

    w_packed, scales, biases = quantize_4bit(w, group_size=group_size, output_device=device)
    y = dequant_matvec(w_packed, scales, biases, x, group_size=group_size)

    w_ref = dequantize_4bit(
        w_packed,
        scales,
        biases,
        in_features=in_features,
        group_size=group_size,
    ).to(device)
    ref = torch.matmul(w_ref, x.to(torch.float32))
    torch.mps.synchronize()

    max_diff = (y.to(torch.float32) - ref).abs().max().item()
    print(f"dequant_matvec max|diff|={max_diff:.3e}")
    return max_diff < 2e-4


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
