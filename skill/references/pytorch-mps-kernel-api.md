# PyTorch MPS Kernel API

## Scope

This guide covers practical use of `torch.mps.compile_shader` and kernel invocation
patterns from Python.

## Compile and Call

```python
lib = torch.mps.compile_shader(source)
lib.my_kernel(tensor_a, tensor_b, out, 128, threads=(out.numel(),), group_size=(256,))
```

## Kernel Signature Rules

- Kernel argument order must match the call argument order.
- Tensor arguments map to `device` buffers.
- Python `int`/`float` become Metal `constant` scalar args.
- Built-in dispatch ids use Metal attributes like `[[thread_position_in_grid]]`.

Example:

```metal
kernel void add_kernel(const device float* x,
                       const device float* y,
                       device float* out,
                       constant uint& n,
                       uint i [[thread_position_in_grid]]) {
  if (i < n) { out[i] = x[i] + y[i]; }
}
```

## Dispatch Controls

- `threads=(n,)` for 1D kernels.
- `threads=(x, y)` or `(x, y, z)` for multi-dim dispatch.
- `group_size=(gx,)` etc controls threads-per-threadgroup.
- Omit `group_size` for runtime default or set explicitly when tuning.

## Argument Casting

Kernel call supports `arg_casts` if scalar widths must be forced.

```python
lib.k(..., 1024, threads=(n,), arg_casts={3: "int32"})
```

Use this when kernel expects `constant int&` and Python would otherwise pass
wider types.

## Device Preconditions

Always gate execution:

```python
if not torch.backends.mps.is_available():
    ...
```

And validate tensors:

- `tensor.device.type == "mps"`
- expected `dtype`
- contiguous layout when index math assumes dense memory

## Runtime Introspection

Each compiled kernel exposes:

- `max_threads_per_threadgroup`
- `thread_execution_width`
- `static_thread_group_memory_length`

Use these values to constrain launch candidates.

## When to Move Beyond compile_shader

Escalate to C++/ObjC++ extension patterns when you need:

- integration with custom autograd kernels in C++
- tighter control over command buffers and synchronization
- registration through `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`
- integration directly into the PyTorch dispatcher (`DispatchKey::MPS`)
