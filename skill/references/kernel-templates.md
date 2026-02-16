# Kernel Templates

## 1D Elementwise Template

```metal
#include <metal_stdlib>
using namespace metal;

kernel void unary_kernel(const device float* x,
                         device float* out,
                         constant uint& n,
                         uint i [[thread_position_in_grid]]) {
  if (i >= n) { return; }
  out[i] = x[i];
}
```

Python call:

```python
lib.unary_kernel(x, out, n, threads=(n,), group_size=(256,))
```

## Row-wise Reduction Template

```metal
kernel void row_reduce(const device float* x,
                       device float* out,
                       constant uint& rows,
                       constant uint& d,
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint3 tg_pos [[threadgroup_position_in_grid]],
                       uint3 tptg [[threads_per_threadgroup]]) {
  uint row = tg_pos.x;
  if (row >= rows) { return; }

  uint tg = tptg.x;
  uint num_sg = (tg + 31) / 32;
  uint base = row * d;

  float local = 0.0f;
  for (uint i = tid; i < d; i += tg) {
    local += x[base + i];
  }
  local = simd_sum(local);

  threadgroup float shared[32];
  if (lane == 0) { shared[sg] = local; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    float partial = (lane < num_sg) ? shared[lane] : 0.0f;
    float total = simd_sum(partial);
    if (lane == 0) { shared[0] = total; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < d; i += tg) {
    out[base + i] = shared[0];
  }
}
```

Dispatch with one threadgroup per row:

```python
group = 256
lib.row_reduce(x, out, rows, d, threads=(rows * group,), group_size=(group,))
```

## 2D Dispatch Template

```metal
kernel void kernel2d(const device float* x,
                     device float* out,
                     constant uint& width,
                     constant uint& height,
                     uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= width || gid.y >= height) { return; }
  uint idx = gid.y * width + gid.x;
  out[idx] = x[idx];
}
```

Python call:

```python
lib.kernel2d(x, out, width, height, threads=(width, height), group_size=(16, 16))
```

## Guardrails

- Always bounds-check when dispatch is rounded up.
- Prefer explicit scalar args (`rows`, `d`, `width`) over deriving from metadata in-kernel.
- Keep barriers under uniform control flow across participating threads.
- Prefer compile-time specialization (for extension paths) over heavy runtime branching.

## Host Launch Validation Template

For reduction and collective kernels, validate launch constraints in host code
before kernel dispatch:

```cpp
const auto tg = group_size;
const auto max_tg = pipeline.maxTotalThreadsPerThreadgroup;
const auto simd = pipeline.threadExecutionWidth;
TORCH_CHECK(tg <= max_tg, "group_size exceeds max threads per threadgroup");
TORCH_CHECK((tg % simd) == 0, "group_size must align with simdgroup width");
```

Add kernel-specific checks (for example divisibility of head size or tile size)
when the kernel assumes exact partitioning.

## Two-Phase Reduction Template (Large Context)

When one pass becomes register/memory bound:

1. Phase A: compute per-partition partials and local normalizers.
2. Phase B: reduce partials into final outputs.

Use separate kernels and intermediate buffers to keep each phase simpler and
easier to tune.

## Function-Constant Variant Template (Extension Path)

For stable feature flags, compile variant pipelines once and cache by key:

```cpp
std::string key = fmt_key(dtype, head_size, use_alibi, use_partitioning);
if (!cache.count(key)) {
  cache[key] = build_pipeline_with_function_constants(...);
}
auto pipeline = cache.at(key);
```

This avoids per-thread branching for optional paths in hot kernels.
