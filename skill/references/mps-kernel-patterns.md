# MPS Kernel Patterns

Use this reference when designing extension-grade MPS kernels or upgrading
existing templates.

## 1. Validate Launch Assumptions on Host Side

Fail fast when launch constraints are invalid (threadgroup size, simdgroup
divisibility, tile assumptions).

Why it matters:

- avoids silent data corruption when collectives assume specific lane layout
- turns hard-to-debug GPU faults into immediate argument errors

## 2. Keep Per-Op Launch Policy in Config

Avoid hardcoding one `group_size`; keep per-op launch knobs in model/runtime
config and pass them into dispatch code.

Why it matters:

- makes tuning explicit and versionable
- allows op-specific defaults without patching kernel code

## 3. Use Simdgroup + Threadgroup Staging for Reductions

Reduction baseline:

1. per-thread local accumulation
2. `simd_sum` within simdgroup
3. one-lane writes to threadgroup scratch
4. final reduction in leading simdgroup

Why it matters:

- good baseline for RMSNorm/LayerNorm/softmax reductions
- predictable correctness when barriers remain uniform

## 4. Function Constants for Variant Specialization

Compile variants with function constants instead of branching on optional
features inside hot kernels (partitioning, ALiBi, fp8 scaling, etc.).

Why it matters:

- lower branch cost in hot loops
- explicit variant cache keyed by runtime features

## 5. Two-Phase Architecture for Large Attention/Reductions

Split into:

1. partition phase (partial stats/output)
2. reduction/finalize phase

Why it matters:

- scales better on long contexts
- easier tuning than one oversized kernel

## 6. Runtime Kernel Selection by Dtype/Shape

Build kernel function names (or pipeline keys) from runtime dtype/layout/head
size where needed.

Why it matters:

- keeps variants explicit
- avoids one monolithic kernel with many dynamic branches

## 7. Embedded Metallib Loading Over Source JIT

Ship metallib blobs and load pipelines from library data in ObjC++.

Why it matters:

- predictable startup behavior
- avoids runtime source compile dependence in production paths

## 8. Integrate with Current PyTorch MPS Stream

Use current stream command buffer/queue and dispatch on that stream.

Why it matters:

- preserves ordering with surrounding PyTorch ops
- avoids hidden sync bugs across command buffers

## 9. Use Blit Encoders for Device-Device Block Moves

For cache paging/block copy operations, use `MTLBlitCommandEncoder` when copy
semantics fit.

Why it matters:

- cleaner and often faster than custom compute copy kernels
- simplifies cache maintenance logic

## 10. Register MPS Ops Through TORCH_LIBRARY(_IMPL)

Define schema once and register MPS backend implementation directly.

Why it matters:

- clean dispatcher integration
- straightforward CPU/CUDA/MPS multi-backend expansion

## 11. Add Cross-Kernel Health Checks and Reporting

Run multi-kernel checks and report failures for unattended/nightly runs.

Why it matters:

- catches regressions across kernels and environments early
- useful as a nightly gate once kernel count grows

## How to Apply in This Skill

1. Keep `compile_shader` scripts as quick prototyping path.
2. Move performance-critical kernels to extension path when specialization needs
   outgrow Python shader dispatch.
3. Require launch validation and variant-key logging in benchmark output.
4. Prefer two-phase kernels for long-context operations.
5. Add CI-style cross-kernel smoke checks as kernel count increases.
