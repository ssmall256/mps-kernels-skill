"""Single-kernel attention pattern for PyTorch MPS.

Demonstrates:
- one threadgroup per (head, query_position)
- online softmax update in-kernel
- optional causal/sliding-window masking
- grouped-query attention (GQA) by mapping q_heads -> kv_heads

This script is intentionally educational. For production workloads, compare
against `torch.nn.functional.scaled_dot_product_attention` and keep the faster
path as default.
"""
from __future__ import annotations

import math
import os
import sys
from typing import TypedDict

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

ATTENTION_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void attention_rowwise(const device float* q,
                              const device float* k,
                              const device float* v,
                              device float* out,
                              constant uint& q_heads,
                              constant uint& kv_heads,
                              constant uint& seq,
                              constant uint& d,
                              constant uint& causal,
                              constant int& window,
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint3 tg_pos [[threadgroup_position_in_grid]],
                              uint3 tptg [[threads_per_threadgroup]]) {
    uint row = tg_pos.x;
    uint total_rows = q_heads * seq;
    if (row >= total_rows) {
        return;
    }

    uint q_head = row / seq;
    uint q_pos = row % seq;
    uint kv_head = (q_head * kv_heads) / q_heads;

    uint tg_size = tptg.x;
    uint num_sg = (tg_size + 31) / 32;
    uint q_base = (q_head * seq + q_pos) * d;
    uint k_head_base = kv_head * seq * d;
    uint v_head_base = kv_head * seq * d;

    // Initialize output row.
    for (uint i = tid; i < d; i += tg_size) {
        out[q_base + i] = 0.0f;
    }

    uint start = 0;
    if (window > 0) {
        int maybe_start = int(q_pos) + 1 - window;
        start = (maybe_start > 0) ? uint(maybe_start) : 0;
    }
    uint end = seq;
    if (causal != 0) {
        end = min(end, q_pos + 1);
    }
    if (start >= end) {
        return;
    }

    float inv_sqrt_d = rsqrt(float(d));
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    threadgroup float shared[32];

    for (uint kv = start; kv < end; ++kv) {
        float dot = 0.0f;
        uint k_base = k_head_base + kv * d;
        for (uint i = tid; i < d; i += tg_size) {
            dot += q[q_base + i] * k[k_base + i];
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            shared[sg] = dot;
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

        float score = shared[0] * inv_sqrt_d;
        float new_max = max(row_max, score);
        float corr = exp(row_max - new_max);
        float exp_score = exp(score - new_max);

        uint v_base = v_head_base + kv * d;
        for (uint i = tid; i < d; i += tg_size) {
            float prev = out[q_base + i];
            out[q_base + i] = prev * corr + exp_score * v[v_base + i];
        }

        row_sum = row_sum * corr + exp_score;
        row_max = new_max;
    }

    float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = tid; i < d; i += tg_size) {
        out[q_base + i] *= inv;
    }
}
"""

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(ATTENTION_SOURCE)
    return _LIB


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: int = 0,
    group_size: int | None = None,
) -> torch.Tensor:
    require_mps()
    if q.device.type != "mps" or k.device.type != "mps" or v.device.type != "mps":
        raise ValueError("q, k, v must be on mps")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be rank-3: (heads, seq, d)")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if q.shape[1] != k.shape[1]:
        raise ValueError("this example currently requires q.shape[1] == k.shape[1]")
    if q.shape[2] != k.shape[2]:
        raise ValueError("q and k must have same head dimension d")

    q_heads, seq, d = map(int, q.shape)
    kv_heads = int(k.shape[0])
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads for GQA mapping")
    if window_size < 0:
        raise ValueError("window_size must be >= 0")

    q32 = q.contiguous().to(torch.float32)
    k32 = k.contiguous().to(torch.float32)
    v32 = v.contiguous().to(torch.float32)
    out32 = torch.empty_like(q32)

    kernel = _lib().attention_rowwise
    max_group, simd_width = kernel_limits(kernel)
    candidates = tuple(g for g in rowwise_group_candidates(max_group) if (g % simd_width) == 0)
    if not candidates:
        raise RuntimeError(
            "attention: no valid group sizes for this kernel "
            f"(max_group={max_group}, simd_width={simd_width})"
        )
    default_group = default_group_size(max_group)
    if default_group not in candidates:
        lower = [g for g in candidates if g <= default_group]
        default_group = max(lower) if lower else candidates[0]

    rows = q_heads * seq

    def _run_with_group(g: int) -> None:
        validate_rowwise_collective_launch(
            op_name="attention",
            rows=rows,
            d=d,
            group_size=int(g),
            max_group=max_group,
            simd_width=simd_width,
        )
        kernel(
            q32,
            k32,
            v32,
            out32,
            q_heads,
            kv_heads,
            seq,
            d,
            1 if causal else 0,
            int(window_size),
            threads=(rows * int(g),),
            group_size=(int(g),),
        )

    if group_size is None:
        group_size = pick_group_size(
            key=f"attention:q={q_heads}:kv={kv_heads}:s={seq}:d={d}:c={int(causal)}:w={window_size}",
            candidates=candidates,
            default=default_group,
            run=_run_with_group,
            warmup=2,
            iters=20,
        )

    _run_with_group(int(group_size))
    return out32.to(dtype=q.dtype)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    window_size: int = 0,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)

    q_heads, seq, d = qf.shape
    kv_heads = int(kf.shape[0])
    group = q_heads // kv_heads
    if group > 1:
        kf = kf.repeat_interleave(group, dim=0)
        vf = vf.repeat_interleave(group, dim=0)

    scale = 1.0 / math.sqrt(float(d))
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale

    if causal:
        causal_mask = torch.triu(
            torch.ones((seq, seq), device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if window_size > 0:
        pos = torch.arange(seq, device=scores.device)
        window_mask = (pos.unsqueeze(1) - pos.unsqueeze(0)) >= window_size
        scores = scores.masked_fill(window_mask.unsqueeze(0), float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)
    return out.to(dtype=q.dtype)


class AttentionCase(TypedDict):
    shape_q: tuple[int, int, int]
    shape_kv: tuple[int, int, int]
    causal: bool
    window_size: int


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping attention validation.")
        return True

    device = torch.device("mps")
    ok = True

    cases: list[AttentionCase] = [
        {"shape_q": (4, 64, 128), "shape_kv": (4, 64, 128), "causal": False, "window_size": 0},
        {"shape_q": (4, 64, 128), "shape_kv": (4, 64, 128), "causal": True, "window_size": 0},
        {"shape_q": (8, 64, 128), "shape_kv": (2, 64, 128), "causal": True, "window_size": 32},
    ]

    for case in cases:
        shape_q = case["shape_q"]
        shape_kv = case["shape_kv"]
        causal = case["causal"]
        window_size = case["window_size"]

        q = torch.randn(*shape_q, device=device, dtype=torch.float32)
        k = torch.randn(*shape_kv, device=device, dtype=torch.float32)
        v = torch.randn(*shape_kv, device=device, dtype=torch.float32)

        y = attention(q, k, v, causal=causal, window_size=window_size)
        ref = _reference_attention(q, k, v, causal=causal, window_size=window_size)
        torch.mps.synchronize()

        max_diff = (y - ref).abs().max().item()
        print(
            "attention",
            f"q={shape_q}",
            f"kv={shape_kv}",
            f"causal={causal}",
            f"window={window_size}",
            f"max|diff|={max_diff:.3e}",
        )
        ok = ok and max_diff < 2e-4

    return ok


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
