"""Benchmark custom RMSNorm kernel against torch.nn.functional.rms_norm."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kernels.utils import benchmark_ms, require_mps
from scripts.rmsnorm_kernel import rmsnorm


def _row(shape: tuple[int, ...], custom_ms: float, ref_ms: float) -> str:
    speedup = ref_ms / custom_ms if custom_ms > 0 else float("inf")
    return f"{str(shape):<18} {custom_ms:>10.3f} {ref_ms:>10.3f} {speedup:>10.2f}"


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping benchmark.")
        return

    require_mps()
    device = torch.device("mps")
    shapes = [
        (1, 2048),
        (8, 4096),
        (4, 512, 4096),
        (2, 1024, 2048),
    ]

    print("shape              custom(ms)    torch(ms)    speedup")
    print("-" * 56)

    for shape in shapes:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32)

        custom_ms = benchmark_ms(lambda: rmsnorm(x, w, eps=1e-5), warmup=3, iters=20)
        ref_ms = benchmark_ms(
            lambda: F.rms_norm(x, (shape[-1],), w, eps=1e-5),
            warmup=3,
            iters=20,
        )

        print(_row(shape, custom_ms, ref_ms))


if __name__ == "__main__":
    main()
