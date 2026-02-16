# Apple Silicon Optimization Guide (PyTorch MPS)

## Practical Heuristics

- Start `group_size` as a multiple of 32.
- Common first candidates: `32, 64, 128, 256`.
- Clamp to `kernel.max_threads_per_threadgroup`.
- Avoid tiny kernels for very small workloads; launch overhead dominates.
- Use one-threadgroup-per-row for row reductions and dispatch as `rows * group_size`.

## Memory and Layout

- Favor contiguous tensors for predictable indexing.
- Prefer fusing adjacent pointwise ops in one kernel when possible.
- Avoid repeated dtype conversions inside tight loops.
- Keep barrier regions under uniform control flow to avoid deadlocks/undefined behavior.

## Numerical Stability

- For reductions, accumulate in float32 even if input/output is lower precision.
- For softmax, subtract row max before exponentiation.
- Compare against reference ops across wide magnitude ranges.

## Benchmark Discipline

- Warm up to populate compilation and cache paths.
- Synchronize after each timed run (`torch.mps.synchronize()`).
- Report shape, dtype, launch settings, and tolerance together.
- Re-check with autotuning enabled (`MPS_KERNELS_AUTOTUNE=1`) before final conclusions.

## Escalation Path

If Python shader kernels are bottlenecked by dispatch overhead or API limits:

1. Move hot paths into a compiled extension.
2. Keep Python wrappers thin and validation-heavy.
3. Register backend-specific implementations through MPS dispatch keys.
