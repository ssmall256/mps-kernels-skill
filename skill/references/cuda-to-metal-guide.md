# CUDA to Metal Guide (PyTorch MPS)

## Mental Model Mapping

- CUDA kernel launch -> `lib.kernel(..., threads=..., group_size=...)`
- CUDA thread block -> Metal threadgroup
- CUDA thread index -> `thread_position_in_grid` / threadgroup indices
- CUDA shared memory -> Metal `threadgroup` memory

## Practical Differences

- Launch controls are passed from Python at call time.
- Scalar argument packing may need explicit casting.
- PyTorch MPS custom-kernel APIs are newer and less feature-complete than mature CUDA extension flows.

## Porting Strategy

1. Port scalar math and indexing first.
2. Validate exact parity with tiny tensors.
3. Introduce launch tuning (`group_size`) after correctness.
4. Add MPS-specific fast paths and keep baseline fallback.

## Dispatch and Registration

For backend integration into operator dispatch, use C++ registration with
`TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL(..., MPS, ...)` patterns.
See `references/extension-patterns.md`.
