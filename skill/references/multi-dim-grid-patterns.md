# Multi-Dimensional Grid Patterns (PyTorch MPS)

## Why It Matters

`torch.mps.compile_shader` supports 1D/2D/3D dispatch through `threads=` and
`group_size=`. Correct grid mapping makes kernels simpler and avoids index bugs.

## 1D Pattern

Use for flat elementwise or rowwise collectives:

- `threads=(rows * group_size,)`
- `group_size=(group_size,)`
- use `threadgroup_position_in_grid.x` as row index

## 2D Pattern

Use for matrix outputs:

- `threads=(width, height)`
- `group_size=(gx, gy)`
- use `uint2 gid [[thread_position_in_grid]]`

Example: `scripts/simdgroup_matmul_kernel.py`.

## 3D Pattern

Use when mapping head/batch dimensions directly:

- `threads=(x, y, z)`
- `group_size=(gx, gy, gz)`
- use `uint3 gid [[thread_position_in_grid]]`

Only use 3D when it improves readability versus flattening.

## Flattening Strategy

For many collectives, flattening is easier:

- flatten `(batch, heads, seq)` into `rows`
- keep `d` as inner dimension
- map row via `threadgroup_position_in_grid.x`

This pattern is robust and works well with autotuned `group_size`.

## Guardrails

- Always bounds-check each axis when dispatch is rounded up.
- Keep threadgroup product `gx * gy * gz` within kernel max threads.
- If using collectives, keep at least one axis aligned to simd width.
