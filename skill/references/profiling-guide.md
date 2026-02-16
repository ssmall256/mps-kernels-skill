# Profiling Guide

## Measurement Baseline

Use synchronized timing:

```python
t0 = time.perf_counter()
run_kernel()
torch.mps.synchronize()
ms = (time.perf_counter() - t0) * 1000
```

## What to Record

- operation name
- input shapes and dtype
- threads/group_size
- p50 latency over N iterations
- reference latency (`torch` baseline)
- warmup count and whether compile time is excluded
- selected variant key (if using function-constant specialization)

## Common Pitfalls

- Timing without synchronization (under-reports work)
- Including first-run compile overhead in steady-state numbers
- Comparing non-equivalent dtypes or tensor layouts
- Reporting one lucky run instead of median
- Mixing partitioned and non-partitioned kernel variants in one aggregate number

## MPS Profiler Hooks

PyTorch exposes an MPS profiler module in `torch.mps.profiler`. Use it for
trace capture when microbenchmarks are not enough.

## Debug Workflow

1. Validate numerical parity first.
2. Benchmark kernel alone and baseline op alone.
3. Sweep group sizes.
4. For two-phase kernels, measure each phase and total time.
5. Inspect end-to-end pipeline impact.
6. Keep only optimizations that survive full-pipeline tests.
