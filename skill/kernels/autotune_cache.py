"""Simple autotune + cache for MPS kernel launch parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable
import json
import os
import threading

from .utils import benchmark_ms

_CACHE_LOCK = threading.Lock()
_MEM_CACHE: Dict[str, int] = {}


def _cache_path() -> Path:
    base = os.environ.get("MPS_KERNELS_CACHE_DIR") or str(
        Path.home() / ".cache" / "mps-kernels-skill"
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / "autotune.json"


def _load_disk() -> Dict[str, int]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_disk(values: Dict[str, int]) -> None:
    path = _cache_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(values, indent=2, sort_keys=True))
    tmp.replace(path)


def autotune_enabled() -> bool:
    v = os.environ.get("MPS_KERNELS_AUTOTUNE", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def pick_group_size(
    *,
    key: str,
    candidates: Iterable[int],
    run: Callable[[int], object],
    default: int,
    warmup: int = 3,
    iters: int = 40,
) -> int:
    """Pick the fastest threads-per-threadgroup candidate and cache it."""
    with _CACHE_LOCK:
        if key in _MEM_CACHE:
            return _MEM_CACHE[key]

        disk = _load_disk()
        if key in disk:
            _MEM_CACHE[key] = int(disk[key])
            return _MEM_CACHE[key]

    if not autotune_enabled():
        with _CACHE_LOCK:
            _MEM_CACHE[key] = int(default)
        return int(default)

    valid = [int(v) for v in candidates if int(v) > 0 and int(v) <= 1024]
    if not valid:
        valid = [int(default)]

    best = valid[0]
    best_ms = float("inf")

    for g in valid:
        try:
            ms = benchmark_ms(lambda: run(g), warmup=warmup, iters=iters)
        except Exception:
            continue
        if ms < best_ms:
            best_ms = ms
            best = g

    with _CACHE_LOCK:
        _MEM_CACHE[key] = int(best)
        disk = _load_disk()
        disk[key] = int(best)
        _save_disk(disk)

    return int(best)
