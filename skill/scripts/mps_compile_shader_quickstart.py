"""Quickstart demo for torch.mps.compile_shader."""
from __future__ import annotations

import torch


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS is not available; skipping quickstart.")
        return

    src = r'''
    #include <metal_stdlib>
    using namespace metal;

    kernel void fill_kernel(device float* out,
                            constant float& value,
                            uint i [[thread_position_in_grid]]) {
        out[i] = value;
    }

    kernel void axpy_kernel(const device float* x,
                            const device float* y,
                            device float* out,
                            constant float& alpha,
                            uint i [[thread_position_in_grid]]) {
        out[i] = alpha * x[i] + y[i];
    }
    '''
    lib = torch.mps.compile_shader(src)

    device = torch.device("mps")
    n = 1024
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    lib.fill_kernel(out, 3.0, threads=(n,), group_size=(256,))
    torch.mps.synchronize()
    print(f"fill_kernel sample: {out[:4].tolist()}")

    lib.axpy_kernel(x, y, out, 0.25, threads=(n,), group_size=(256,))
    torch.mps.synchronize()

    ref = 0.25 * x + y
    max_diff = (out - ref).abs().max().item()
    print(f"axpy max|diff|={max_diff:.3e}")


if __name__ == "__main__":
    main()
