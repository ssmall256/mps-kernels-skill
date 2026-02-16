# Metal 4 Spec Notes for PyTorch MPS Kernel Work

Source reviewed: Metal Shading Language Specification v4 (2025-10-23).

## Why This Matters

Metal 4 expands the language and ML-oriented primitives. This reference maps those
additions to practical decisions in this skill:

- use now in `torch.mps.compile_shader`
- use in extension path (`.mm` + custom runtime)
- defer until framework support is clear

## High-Impact Metal 4 Additions

## C++17 Base Language

Metal 4 is C++17-based with Metal-specific restrictions and address-space rules.
Practical impact:

- favor explicit address-space qualifiers on pointers/references
- keep kernel helper templates simple and address-space aware
- avoid unsupported C++ features in shader code paths

## Tensor and Cooperative Tensor Types

MSL 4 introduces `tensor<...>` and `cooperative_tensor<...>` plus tensor operations
in Metal Performance Primitives (MPP).

Practical impact:

- this is relevant for matmul/conv-heavy kernels and structured reductions
- treat as advanced/extension-first unless proven in your exact PyTorch runtime path
- keep existing pointer-based kernels as default templates for broad compatibility

## SIMD-Group Matrix APIs

The `<metal_simdgroup_matrix>` APIs provide matrix load/store and MMA-style ops.
Practical impact:

- useful for high-throughput small-tile matrix blocks
- requires careful tiling and hardware-aware kernel design
- good candidate for extension path or dedicated experimental scripts

## Uniform Type and Uniform Control Flow

`uniform<T>` can encode values that are identical for all threads in relevant scope.
Practical impact:

- structure barrier and cooperative sections under uniform control flow
- avoid divergent control around `threadgroup_barrier` / `simdgroup_barrier`
- this helps correctness and can improve compiler optimization opportunities

## Function Constants

Function constants provide compile-time specialization without source duplication.
Practical impact:

- specialize block sizes, optional paths, and resource-index patterns
- reduce runtime branching in hot kernels
- best suited to extension pipelines or controlled shader compilation flows

## Barrier and Memory-Coherency Discipline

The spec reiterates strict barrier rules: all participating threads must execute
matching barriers in the same dynamic pattern.

Practical impact:

- do not place barriers under divergent branches
- maintain one-threadgroup-per-row patterns with consistent control flow
- pair barrier usage with explicit memory-space assumptions

## Recommended Skill Routing

Use this sequence when users ask about "new Metal 4" ideas:

1. Validate if the feature is required for performance/correctness.
2. Check if a pointer-based kernel template already solves the need.
3. If not, choose extension path and prototype there first.
4. Benchmark against baseline ops and keep only proven wins.

## Integration Checklist for New Kernels

- Confirm required Metal version and GPU support.
- Decide if `compile_shader` is sufficient or extension path is needed.
- Keep control flow barrier-safe and preferably uniform around cooperative sections.
- Prefer compile-time specialization over runtime branching for stable hot paths.
- Add parity tests and synchronized benchmarks before promoting defaults.

## Notes on Scope in This Skill

This skill currently treats the following as primary defaults:

- pointer-based `torch.mps.compile_shader` kernels
- row-wise/threadgroup reduction patterns
- extension migration when API limits are reached

This keeps examples runnable while still documenting where Metal 4 can unlock
higher-performance designs.
