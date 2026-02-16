"""Small utilities for PyTorch MPS kernel scripts.

This module stays defensive so examples run cleanly across multiple PyTorch versions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import time

import torch


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    is_available: bool
    built_with_mps: bool


def mps_is_available() -> bool:
    return bool(torch.backends.mps.is_available())


def mps_is_built() -> bool:
    return bool(getattr(torch.backends.mps, "is_built", lambda: False)())


def get_device_info() -> DeviceInfo:
    available = mps_is_available()
    built = mps_is_built()
    if not available:
        if not built:
            name = "MPS not built into this PyTorch"
        else:
            name = "MPS unavailable on this machine/runtime"
        return DeviceInfo(name=name, is_available=False, built_with_mps=built)

    name = "Apple MPS"
    get_name = getattr(torch._C, "_mps_get_name", None)
    if callable(get_name):
        try:
            name = str(get_name())
        except Exception:
            pass
    return DeviceInfo(name=name, is_available=True, built_with_mps=built)


def require_mps() -> torch.device:
    if not mps_is_available():
        raise RuntimeError(
            "MPS is not available. Run on macOS with Apple Silicon and an MPS-enabled PyTorch build."
        )
    return torch.device("mps")


def synchronize_if_needed(device: Optional[torch.device] = None) -> None:
    dev = device.type if isinstance(device, torch.device) else "mps"
    if dev == "mps" and mps_is_available():
        torch.mps.synchronize()


def benchmark_ms(
    fn: Callable[[], object],
    *,
    warmup: int = 5,
    iters: int = 50,
    device: Optional[torch.device] = None,
) -> float:
    for _ in range(max(0, warmup)):
        _ = fn()
        synchronize_if_needed(device)

    times = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        _ = fn()
        synchronize_if_needed(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def as_contiguous_2d_lastdim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """Flatten to (rows, D) with contiguous storage and return shape metadata."""
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    orig_shape = tuple(x.shape)
    d = int(x.shape[-1])
    rows = int(x.numel() // d)
    x2 = x.reshape(rows, d).contiguous()
    return x2, orig_shape, d


def restore_from_2d(x2: torch.Tensor, orig_shape: tuple[int, ...]) -> torch.Tensor:
    return x2.reshape(orig_shape)


def rowwise_group_candidates(max_group: int) -> tuple[int, ...]:
    if max_group < 1:
        return (1,)
    if max_group < 32:
        return (max_group,)

    base = [32, 64, 128, 256, 512, 1024]
    values = [g for g in base if g <= max_group]
    rounded = (max_group // 32) * 32
    if rounded >= 32 and rounded not in values:
        values.append(rounded)
    values.sort()
    return tuple(values)


def default_group_size(max_group: int, preferred: int = 256) -> int:
    candidates = rowwise_group_candidates(max_group)
    valid = [g for g in candidates if g <= preferred]
    if valid:
        return max(valid)
    return candidates[-1]


def kernel_limits(kernel: object, *, default_simd: int = 32) -> tuple[int, int]:
    """Return (max_threads_per_threadgroup, thread_execution_width)."""
    max_group = int(getattr(kernel, "max_threads_per_threadgroup", 1024))
    simd_width = int(getattr(kernel, "thread_execution_width", default_simd))
    if max_group < 1:
        raise ValueError(f"invalid kernel max_threads_per_threadgroup={max_group}")
    if simd_width < 1:
        raise ValueError(f"invalid kernel thread_execution_width={simd_width}")
    return max_group, simd_width


def validate_threadgroup_size(
    *,
    op_name: str,
    group_size: int,
    max_group: int,
    simd_width: int | None = None,
    require_simd_multiple: bool = False,
) -> None:
    if group_size < 1:
        raise ValueError(f"{op_name}: group_size must be >= 1, got {group_size}")
    if group_size > max_group:
        raise ValueError(
            f"{op_name}: group_size={group_size} exceeds max_threads_per_threadgroup={max_group}"
        )
    if require_simd_multiple:
        if simd_width is None or simd_width < 1:
            raise ValueError(f"{op_name}: valid simd_width required for collective kernel")
        if (group_size % simd_width) != 0:
            raise ValueError(
                f"{op_name}: group_size={group_size} must be divisible by thread_execution_width={simd_width}"
            )


def validate_rowwise_collective_launch(
    *,
    op_name: str,
    rows: int,
    d: int,
    group_size: int,
    max_group: int,
    simd_width: int,
    max_simdgroups: int = 32,
) -> None:
    if rows < 1:
        raise ValueError(f"{op_name}: rows must be >= 1, got {rows}")
    if d < 1:
        raise ValueError(f"{op_name}: d must be >= 1, got {d}")

    validate_threadgroup_size(
        op_name=op_name,
        group_size=group_size,
        max_group=max_group,
        simd_width=simd_width,
        require_simd_multiple=True,
    )

    num_sg = (group_size + simd_width - 1) // simd_width
    if num_sg > max_simdgroups:
        raise ValueError(
            f"{op_name}: group_size={group_size} implies {num_sg} simdgroups, "
            f"exceeding shared scratch capacity ({max_simdgroups})"
        )
