"""Very small smoke tests for local runs and CI.

Run:
  python -m skill.tests.smoke_test
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _run_snippet(snippet: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def _mps_is_usable() -> bool:
    try:
        import torch
    except Exception as exc:
        print("torch import failed; skipping smoke tests:", exc)
        return False

    built = bool(getattr(torch.backends.mps, "is_built", lambda: False)())
    available = bool(torch.backends.mps.is_available())
    if available:
        return True

    if not built:
        print("MPS backend is not built in this torch install; skipping smoke tests.")
    else:
        print("MPS backend is built but unavailable in this runtime; skipping smoke tests.")
    return False


def _mps_required() -> bool:
    value = os.environ.get("MPS_REQUIRED", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def test_rmsnorm() -> None:
    snippet = textwrap.dedent(
        """
        import torch
        from skill.scripts.rmsnorm_kernel import rmsnorm

        x = torch.randn(2, 128, 256, device='mps', dtype=torch.float32)
        w = torch.randn(256, device='mps', dtype=torch.float32)

        y = rmsnorm(x, w, eps=1e-5)
        y_ref = torch.nn.functional.rms_norm(x, (256,), w, eps=1e-5)
        torch.mps.synchronize()

        diff = (y - y_ref).abs().max().item()
        if diff > 1e-5:
            raise AssertionError(f'rmsnorm max|diff|={diff}')
        """
    )
    rc, output = _run_snippet(snippet)
    if rc != 0:
        raise AssertionError(f"rmsnorm smoke test failed:\n{output}")


def test_softmax() -> None:
    snippet = textwrap.dedent(
        """
        import torch
        from skill.scripts.softmax_kernel import softmax

        x = torch.randn(8, 257, device='mps', dtype=torch.float32)
        y = softmax(x)
        y_ref = torch.softmax(x, dim=-1)
        torch.mps.synchronize()

        diff = (y - y_ref).abs().max().item()
        sum_err = (y.sum(dim=-1) - 1.0).abs().max().item()
        if diff > 1e-5 or sum_err > 1e-5:
            raise AssertionError(f'softmax diff={diff} sum_err={sum_err}')
        """
    )
    rc, output = _run_snippet(snippet)
    if rc != 0:
        raise AssertionError(f"softmax smoke test failed:\n{output}")


def main() -> None:
    if not _mps_is_usable():
        if _mps_required():
            raise SystemExit("MPS_REQUIRED is set, but MPS is unavailable in this runtime.")
        return
    test_rmsnorm()
    test_softmax()
    print("smoke tests passed")


if __name__ == "__main__":
    main()
