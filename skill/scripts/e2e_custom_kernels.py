"""End-to-end model patching example for custom MPS kernels.

This script demonstrates replacing `nn.RMSNorm` modules with a custom
`torch.mps.compile_shader`-backed implementation from `rmsnorm_kernel.py`.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.rmsnorm_kernel import rmsnorm


def _benchmark_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(max(0, warmup)):
        _ = fn()
        torch.mps.synchronize()

    times = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        _ = fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


class CustomRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, eps=self.eps)


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.norm = nn.RMSNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.norm(y)
        y = torch.nn.functional.silu(y)
        return self.fc2(y)


def patch_rmsnorm_modules(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            dim = int(child.weight.numel())
            eps = 1e-5 if child.eps is None else float(child.eps)
            replacement = CustomRMSNorm(dim, eps=eps)
            replacement = replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += patch_rmsnorm_modules(child)
    return replaced


def validate_and_benchmark() -> bool:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping e2e custom kernel example.")
        return True

    torch.manual_seed(0)
    device = torch.device("mps")
    dim = 1024
    batch = 8
    seq = 256

    baseline = TinyBlock(dim).to(device=device, dtype=torch.float32).eval()
    patched = TinyBlock(dim).to(device=device, dtype=torch.float32).eval()
    patched.load_state_dict(baseline.state_dict())
    replaced = patch_rmsnorm_modules(patched)
    patched.eval()

    x = torch.randn(batch, seq, dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        y_base = baseline(x)
        y_patch = patched(x)
        torch.mps.synchronize()
    diff = (y_base - y_patch).abs().max().item()
    print(f"patched rmsnorm modules: {replaced}")
    print(f"e2e max|diff|={diff:.3e}")

    with torch.no_grad():
        base_ms = _benchmark_ms(lambda: baseline(x))
        patch_ms = _benchmark_ms(lambda: patched(x))

    print("")
    print("=== e2e custom kernel benchmark ===")
    print(f"{'model':<16} {'p50 ms':>10}")
    print("-" * 28)
    print(f"{'baseline':<16} {base_ms:>10.3f}")
    print(f"{'patched':<16} {patch_ms:>10.3f}")
    print("-" * 28)

    return diff < 1e-4


if __name__ == "__main__":
    ok = validate_and_benchmark()
    print("PASS" if ok else "FAIL")
