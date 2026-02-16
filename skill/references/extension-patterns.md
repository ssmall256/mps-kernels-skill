# Extension Patterns (C++ / ObjC++)

## When to Use

Use this path when `torch.mps.compile_shader` is not enough for your kernel or
when you need dispatcher-integrated ops.

## Build Skeleton

`torch.utils.cpp_extension.load(...)` can compile `.mm` files and expose
bindings back to Python.

Typical pieces:

1. `custom_kernel.mm` with `torch::Tensor` entry points
2. Metal kernels (`.metal` or embedded source)
3. Command-buffer dispatch via PyTorch MPS hooks
4. Python wrapper module for validation and benchmarks

## Community-Proven Runtime Pattern

For MPS extension code, prefer working from the active PyTorch MPS stream:

- get current stream via `at::mps::getCurrentMPSStream()`
- use stream `commandBuffer()` and `queue()` for command submission
- use `dispatch_sync` around encoder construction when required by the stream model

This pattern is used in local MPS kernel repos and keeps extension dispatch aligned
with PyTorch stream ordering.

## Metal 4-Oriented Opportunities

For extension work, consider Metal 4 concepts where they clearly help:

- function-constant specialization for block sizes and optional branches
- `simdgroup_matrix` APIs for tiled matrix math
- tensor/cooperative-tensor pipelines when integrating with MPP-style operators

Treat these as opt-in advanced paths and keep baseline pointer-kernel fallbacks.

## Function-Constant Specialization Pattern

If one op has several stable variants (for example: partitioned vs non-partitioned,
ALiBi on/off, fp8 scaling on/off), compile one function per variant with function
constants, then dispatch by variant key at runtime.

This avoids hot-path branching in-kernel and keeps fast paths explicit.

## Dispatcher Registration

Use:

- `TORCH_LIBRARY(namespace, m)` for schema definition
- `TORCH_LIBRARY_IMPL(namespace, MPS, m)` for MPS backend implementation

This keeps CPU/CUDA/MPS variants structured under one operator interface.

## MPS Runtime Hooks

In ObjC++ extension code, common hooks include:

- `torch::mps::get_command_buffer()`
- `torch::mps::get_dispatch_queue()`
- `torch::mps::commit()`

See local references in:

- your extension's ObjC++ dispatch bridge (`*.mm`) where MPS command encoding happens
- PyTorch MPS backend sources under `aten/src/ATen/native/mps/`
- PyTorch MPS module bridge under `torch/csrc/mps/Module.cpp`

## Migration Pattern

1. Prototype in `compile_shader` script and lock correctness.
2. Port kernel to extension while preserving test vectors.
3. Add dispatcher registration and Python API wrapper.
4. Run parity + performance suites before switching defaults.

## Barrier and Control-Flow Discipline

When using threadgroup/simdgroup collectives, ensure all participating threads
execute matching barriers in the same dynamic pattern. Avoid divergent control
flow around barrier regions.

## Launch Validation Checklist (Host Side)

Before encoding dispatches, validate:

1. `group_size <= maxTotalThreadsPerThreadgroup`
2. simdgroup-collective kernels use a `group_size` compatible with
   `threadExecutionWidth` assumptions
3. reduction tile dimensions are divisible where the kernel requires it
4. all tensors are contiguous/aligned as expected by pointer arithmetic

Fail fast with clear messages when checks do not hold.

## Two-Phase Kernel Architecture

For long-context attention and large reductions, split work into:

1. partition pass: local statistics/output fragments per partition
2. reduce pass: finalize over partition outputs

This pattern improves scalability and keeps each kernel simpler than a giant
single-pass implementation.
