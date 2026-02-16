# SIMDGroup Matrix Guide (PyTorch MPS Context)

## Scope

`simdgroup_matrix` MMA operations are Metal features aimed at tiled matrix math.
In this skill, treat them as **extension-first** capabilities.

## Practical Routing

1. Prototype matmul/fused math in `compile_shader` first.
2. If bottleneck remains matrix-core limited, move to extension path.
3. Implement MMA-specialized kernels with function-constant variants.

## Why Extension-First

- tighter control over pipeline specialization and metallib loading
- cleaner integration with stream/command-buffer management
- better fit for advanced kernel-library style implementations

## Validation Expectations

For MMA migrations:

- preserve reference parity with `torch.matmul` / fused baseline
- benchmark tile-size variants and keep architecture-specific winners
- keep fallback kernel path for unsupported devices/configs

## Related References

- `references/extension-patterns.md`
- `references/metal-4-spec-notes.md`
- `scripts/simdgroup_matmul_kernel.py` (dispatch/reference baseline)
