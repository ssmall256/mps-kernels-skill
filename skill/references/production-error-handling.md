# Production Error Handling (PyTorch MPS Kernels)

## Device and Build Checks

Fail early when MPS is unusable:

- `torch.backends.mps.is_built()`
- `torch.backends.mps.is_available()`

If unavailable, route to a CPU fallback or a baseline torch op.

## Input Validation

Before dispatch:

1. validate rank/layout assumptions
2. validate dtype/device
3. validate shape compatibility
4. validate launch constraints (`group_size`, simd alignment)

Return clear error messages that include the offending values.

## Runtime Safety Pattern

- keep bounds checks in kernel for rounded dispatches
- avoid divergent control around barriers
- keep host-side launch checks as strict as possible

## Fallback Strategy

Recommended pattern per op:

1. try custom kernel path
2. on validation/runtime failure, log context
3. execute baseline torch path
4. optionally disable custom path for the remainder of process lifetime

## Observability

Record per-op diagnostics:

- op name + shape + dtype
- selected group size and variant key
- max error vs baseline in canary mode
- p50 latency and fallback count

## Deployment Guidance

- ship custom kernels behind feature flags
- ramp gradually by workload and tensor-shape buckets
- keep golden correctness tests in CI and pre-release checks
