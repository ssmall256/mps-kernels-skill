---
name: mps-kernels
description: "Guidance for writing custom Metal compute kernels for PyTorch on Apple Silicon using torch.mps.compile_shader and MPS C++ extensions. Covers kernel patterns (elementwise, row-wise normalization/reductions including RMSNorm, LayerNorm, softmax), benchmarking, debugging, and migration paths to TORCH_LIBRARY_IMPL/DispatchKey::MPS when Python shader APIs are not enough."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: elementwise, reduction, rmsnorm, softmax, benchmark, profiling, debugging, extension, dispatch, cuda-porting, metal4, cooperative-tensor, simdgroup-matrix, function-constants, uniform-control-flow, mps-patterns, attention, attention-variants, causal-attention, gqa, rope, quantized, e2e-integration, metallib-embedding"
---

# PyTorch MPS Kernels for Apple Silicon

## Routing Guide (Pick the Right Template)

- **First custom kernel in Python** -> `scripts/mps_compile_shader_quickstart.py`
- **Elementwise / pointwise ops** -> `scripts/batch_elementwise_kernel.py`
- **Row-wise normalization/reductions (RMSNorm / LayerNorm / Softmax)** -> `scripts/rmsnorm_kernel.py`, `scripts/layernorm_kernel.py`, `scripts/softmax_kernel.py`
- **Attention kernels (single + variants)** -> `scripts/attention_kernel.py`, `scripts/attention_variants_kernel.py`
- **RoPE (multi-head)** -> `scripts/multihead_rope_kernel.py`
- **Quantized dequant-matvec pattern** -> `scripts/dequant_matvec_kernel.py`
- **Microbenchmarks** -> `scripts/benchmark_rmsnorm.py`, `scripts/bench_all.py`
- **End-to-end model patching demo** -> `scripts/e2e_custom_kernels.py`
- **Metal 4 feature triage** -> `references/metal-4-spec-notes.md`
- **Need C++/ObjC++ extension path** -> `references/extension-patterns.md`
- **MPS kernel design patterns** -> `references/mps-kernel-patterns.md`

## Pre-Benchmark Checklist (Avoid Timing Lies)

1. Always call `torch.mps.synchronize()` around timing windows.
2. Ignore first-run numbers (shader compile + cache warming).
3. Ensure inputs are contiguous when indexing assumes dense layout.
4. Keep dtypes explicit (cast once, not inside hot loops).
5. Sweep `group_size` and cache best values.

## When This Skill Applies

- Target backend is **PyTorch on Apple Silicon MPS**.
- You need custom behavior not available in `torch.*` or want to fuse operations.
- You are deciding between **Python shader API** and **C++ extension**.
- You need repeatable correctness + benchmark workflows for Metal kernels.

## When NOT to Write Custom Kernels

- A native `torch` op already does the job and is fast enough.
- Tensor sizes are too small for kernel launch overhead to amortize.
- You do not yet have a correctness baseline and tolerance criteria.

## `torch.mps.compile_shader()` Quick API Notes

- Compile once: `lib = torch.mps.compile_shader(source)`.
- Invoke kernel: `lib.kernel_name(*args, threads=(...), group_size=(...))`.
- `threads` may be 1D/2D/3D. If omitted, it defaults to `numel(first_tensor_arg)`.
- Scalars/lists are legal arguments; use `arg_casts` when the kernel expects specific scalar widths.
- Check runtime limits via kernel properties:
  - `max_threads_per_threadgroup`
  - `thread_execution_width`

See `references/pytorch-mps-kernel-api.md` for details and edge cases.

## Core Workflow

1. Start from the closest script template.
2. Add strict shape/dtype/device checks in Python wrapper.
3. Validate against a PyTorch reference implementation.
4. Add synchronized benchmarks and launch-parameter notes.
5. If Python shader path is limiting, escalate to C++ extension patterns.

## Reference Documents

- `references/pytorch-mps-kernel-api.md` - compile_shader call model, arguments, dispatch semantics
- `references/kernel-templates.md` - copy-paste template kernels (elementwise, row-wise, 2D)
- `references/apple-silicon-optimization-guide.md` - launch and memory heuristics for Apple GPUs
- `references/profiling-guide.md` - measurement workflow and profiler usage
- `references/troubleshooting.md` - common failures and fixes
- `references/testing-patterns.md` - regression and edge-case testing strategy
- `references/cuda-to-metal-guide.md` - CUDA to Metal mental model mapping
- `references/extension-patterns.md` - C++ extension + `DispatchKey::MPS` migration path
- `references/metal-4-spec-notes.md` - practical mapping of MSL 4 features to this skill
- `references/mps-kernel-patterns.md` - proven MPS kernel patterns for extension-grade implementations
- `references/attention-kernel-guide.md` - rowwise attention and online softmax kernel structure
- `references/attention-variants-guide.md` - causal, windowed, and GQA variant mapping
- `references/multi-dim-grid-patterns.md` - 1D/2D/3D dispatch patterns in `compile_shader`
- `references/quantized-kernel-patterns.md` - packed 4-bit + fused dequant matvec strategy
- `references/production-error-handling.md` - rollout/fallback patterns for production use
- `references/kernel-debugging-walkthrough.md` - practical checklist for narrowing kernel bugs
- `references/simdgroup-matrix-guide.md` - routing guidance for matrix-core extension paths
- `references/pytorch-integration.md` - model/module integration and patching strategy
- `references/torch-compile-interaction.md` - how `torch.compile` interacts with custom kernels

## Example Scripts

- `scripts/mps_compile_shader_quickstart.py` - tiny fill and AXPY kernels
- `scripts/batch_elementwise_kernel.py` - SiLU pointwise kernel with optional autotune
- `scripts/rmsnorm_kernel.py` - row-wise RMSNorm kernel + validation
- `scripts/layernorm_kernel.py` - row-wise LayerNorm kernel + validation
- `scripts/softmax_kernel.py` - row-wise stable softmax kernel + validation
- `scripts/benchmark_rmsnorm.py` - side-by-side custom vs `torch.nn.functional.rms_norm`
- `scripts/bench_all.py` - quick benchmark suite across included kernels
- `scripts/attention_kernel.py` - rowwise attention kernel with online softmax
- `scripts/attention_variants_kernel.py` - causal, sliding-window, and GQA wrappers
- `scripts/multihead_rope_kernel.py` - multi-head RoPE for `(B, H, S, D)` tensors
- `scripts/dequant_matvec_kernel.py` - packed 4-bit dequantized matvec kernel
- `scripts/simdgroup_matmul_kernel.py` - matrix-dispatch baseline + extension routing note
- `scripts/e2e_custom_kernels.py` - patch `nn.RMSNorm` in a toy model and benchmark

## Known Quirks

- `compile_shader` is available only in MPS-enabled PyTorch builds.
- Kernel argument dtype mismatches can fail at runtime with opaque Metal errors.
- First invocation includes compile overhead; cache behavior can dominate short runs.
