# Kernel Debugging Walkthrough

## 1. Reproduce on the Smallest Shape

Start with minimal deterministic shapes and fixed seed.
Avoid debugging on full model shapes first.

## 2. Isolate the Phase

For multi-stage kernels, check one phase at a time:

1. reduction terms
2. normalization/scaling terms
3. writeback path

## 3. Shrink Launch Complexity

Debug with smaller `group_size` first (for example, one simdgroup).
If one-simdgroup works and multi-simdgroup fails, focus on shared-memory
staging and barrier placement.

## 4. Verify Index Math Explicitly

For each thread id, confirm:

- row index mapping
- inner-dimension stride loop
- base offsets into flat buffers

Common issues are off-by-one in flattened row/head indexing.

## 5. Compare Against Reference Incrementally

Compare intermediate values, not only final output:

- row max / row sum
- mean / variance terms
- per-row dot products

## 6. Validate Launch Constraints

On host side, confirm:

- `group_size <= max_threads_per_threadgroup`
- collectives use group sizes divisible by simd width
- scratch-memory assumptions match `num_sg`

## 7. Re-run Accuracy Sweep

After a fix, sweep:

- multiple shapes (including odd sizes)
- multiple dtypes
- edge masks (causal/window)

Keep the regression case in tests.
