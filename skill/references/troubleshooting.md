# Troubleshooting

## "MPS is not available"

- Confirm PyTorch build includes MPS (`torch.backends.mps.is_built()`).
- Confirm runtime availability (`torch.backends.mps.is_available()`).
- Run on macOS 12.3+ with Apple Silicon GPU.

## Metal Compile Errors

- Check kernel signature and Python argument order match exactly.
- Verify scalar widths (`int32` vs `int64`) and use `arg_casts` if required.
- Ensure all Metal syntax includes necessary headers and `using namespace metal;`.

## Runtime Misbehavior / Wrong Results

- Add bounds checks to every kernel path.
- Force contiguous input when indexing assumes packed layout.
- Validate with very small test tensors and hand-checkable outputs.
- Compare against `torch` reference op and print max absolute error.

## Performance Regressions

- Confirm synchronization is used in timing loops.
- Separate first-run from steady-state.
- Test multiple `group_size` values.
- Verify kernel math is not accidentally serializing too much work.

## Hard-to-Debug Extension Failures

- Start with `torch.mps.compile_shader` first to validate kernel logic.
- Then move to C++ extension and keep the same test vectors.
- Isolate command-buffer setup and argument packing in helper functions.
