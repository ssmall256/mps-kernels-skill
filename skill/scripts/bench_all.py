"""Run a quick benchmark suite for PyTorch MPS kernel examples."""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.utils import benchmark_ms
from scripts.batch_elementwise_kernel import batch_silu
from scripts.layernorm_kernel import layernorm
from scripts.rmsnorm_kernel import rmsnorm
from scripts.softmax_kernel import softmax


def _row(name: str, dtype: str, shape: tuple[int, ...], ms: float) -> str:
    return f"{name:<20} {dtype:<10} {str(shape):<18} {ms:>8.3f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick MPS kernel benchmarks.")
    parser.add_argument(
        "--show-launch",
        action="store_true",
        help="Print resolved launch details for rmsnorm, layernorm, and softmax.",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Include advanced kernels (attention, RoPE, dequant matvec, matmul).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping benchmark suite.")
        return

    device = torch.device("mps")
    print("=== mps-kernels-skill: quick bench ===")
    print(f"{'kernel':<20} {'dtype':<10} {'shape':<18} {'p50 ms':>8}")
    print("-" * 62)

    x = torch.randn(2, 8, 4096, device=device, dtype=torch.float32)
    w = torch.ones(4096, device=device, dtype=torch.float32)

    ms = benchmark_ms(lambda: rmsnorm(x, w, eps=1e-5), warmup=3, iters=20)
    print(_row("rmsnorm", "float32", tuple(x.shape), ms))
    if args.show_launch:
        _, launch = rmsnorm(x, w, eps=1e-5, return_launch_config=True)
        print(
            f"  rmsnorm launch: group_size={launch.group_size} "
            f"simd_width={launch.simd_width}"
        )

    b = torch.zeros(4096, device=device, dtype=torch.float32)
    ms = benchmark_ms(lambda: layernorm(x, w, b, eps=1e-5), warmup=3, iters=20)
    print(_row("layernorm", "float32", tuple(x.shape), ms))
    if args.show_launch:
        _, launch = layernorm(x, w, b, eps=1e-5, return_launch_config=True)
        print(
            f"  layernorm launch: group_size={launch.group_size} "
            f"simd_width={launch.simd_width}"
        )

    x2 = x.reshape(-1, x.shape[-1])
    ms = benchmark_ms(lambda: softmax(x2), warmup=3, iters=20)
    print(_row("softmax", "float32", tuple(x2.shape), ms))
    if args.show_launch:
        _, launch = softmax(x2, return_launch_config=True)
        print(
            f"  softmax launch: variant={launch.variant} "
            f"group_size={launch.group_size} simd_width={launch.simd_width}"
        )

    ms = benchmark_ms(lambda: batch_silu(x), warmup=3, iters=20)
    print(_row("batch_silu", "float32", tuple(x.shape), ms))

    if args.advanced:
        from scripts.attention_kernel import attention
        from scripts.dequant_matvec_kernel import dequant_matvec, quantize_4bit
        from scripts.multihead_rope_kernel import multihead_rope
        from scripts.simdgroup_matmul_kernel import matmul_2d

        q = torch.randn(8, 256, 128, device=device, dtype=torch.float32)
        k = torch.randn(8, 256, 128, device=device, dtype=torch.float32)
        v = torch.randn(8, 256, 128, device=device, dtype=torch.float32)
        ms = benchmark_ms(lambda: attention(q, k, v, causal=True), warmup=2, iters=10)
        print(_row("attn_causal", "float32", tuple(q.shape), ms))

        xrope = torch.randn(2, 8, 256, 128, device=device, dtype=torch.float32)
        ms = benchmark_ms(lambda: multihead_rope(xrope, offset=0), warmup=2, iters=10)
        print(_row("rope_mh", "float32", tuple(xrope.shape), ms))

        w = torch.randn(1024, 1024, device=device, dtype=torch.float32)
        xv = torch.randn(1024, device=device, dtype=torch.float32)
        w_packed, scales, biases = quantize_4bit(w, group_size=64, output_device=device)
        ms = benchmark_ms(
            lambda: dequant_matvec(w_packed, scales, biases, xv, group_size=64),
            warmup=2,
            iters=10,
        )
        print(_row("dequant_matvec", "float32", (1024, 1024), ms))

        a = torch.randn(256, 256, device=device, dtype=torch.float32)
        b2 = torch.randn(256, 256, device=device, dtype=torch.float32)
        ms = benchmark_ms(lambda: matmul_2d(a, b2), warmup=2, iters=10)
        print(_row("matmul_2d", "float32", (256, 256), ms))

    print("-" * 62)


if __name__ == "__main__":
    main()
