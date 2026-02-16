"""Attention variants (causal, sliding-window, and GQA) for PyTorch MPS.

This builds on `attention_kernel.py` and exposes explicit wrappers for common
attention masking/layout variants.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.attention_kernel import _reference_attention, attention


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int | None = None,
) -> torch.Tensor:
    return attention(q, k, v, causal=True, window_size=0, group_size=group_size)


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    group_size: int | None = None,
) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    return attention(q, k, v, causal=False, window_size=window_size, group_size=group_size)


def grouped_query_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    window_size: int = 0,
    group_size: int | None = None,
) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q, k, v must be rank-3: (heads, seq, d)")
    if int(q.shape[0]) % int(k.shape[0]) != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    return attention(
        q,
        k,
        v,
        causal=causal,
        window_size=window_size,
        group_size=group_size,
    )


def validate() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping attention variants validation.")
        return True

    device = torch.device("mps")
    ok = True

    # Causal attention
    q = torch.randn(4, 64, 128, device=device, dtype=torch.float32)
    k = torch.randn(4, 64, 128, device=device, dtype=torch.float32)
    v = torch.randn(4, 64, 128, device=device, dtype=torch.float32)
    y = causal_attention(q, k, v)
    ref = _reference_attention(q, k, v, causal=True, window_size=0)
    torch.mps.synchronize()
    diff = (y - ref).abs().max().item()
    print(f"causal max|diff|={diff:.3e}")
    ok = ok and diff < 2e-4

    # Sliding window
    y = sliding_window_attention(q, k, v, window_size=32)
    ref = _reference_attention(q, k, v, causal=False, window_size=32)
    torch.mps.synchronize()
    diff = (y - ref).abs().max().item()
    print(f"sliding_window max|diff|={diff:.3e}")
    ok = ok and diff < 2e-4

    # GQA + causal
    q = torch.randn(8, 64, 128, device=device, dtype=torch.float32)
    k = torch.randn(2, 64, 128, device=device, dtype=torch.float32)
    v = torch.randn(2, 64, 128, device=device, dtype=torch.float32)
    y = grouped_query_attention(q, k, v, causal=True)
    ref = _reference_attention(q, k, v, causal=True, window_size=0)
    torch.mps.synchronize()
    diff = (y - ref).abs().max().item()
    print(f"gqa_causal max|diff|={diff:.3e}")
    ok = ok and diff < 2e-4

    return ok


if __name__ == "__main__":
    passed = validate()
    print("PASS" if passed else "FAIL")
