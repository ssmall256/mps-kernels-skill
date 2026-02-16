# PyTorch Integration Guide

## Goal

Integrate custom MPS kernels into model code without sacrificing maintainability.

## Module-Level Pattern

1. keep kernel wrappers in `skill/scripts/` during prototyping
2. wrap selected kernels in `nn.Module` adapters
3. patch target modules (for example `nn.RMSNorm`) in one place
4. benchmark baseline vs patched end-to-end

Reference script:

- `scripts/e2e_custom_kernels.py`

## Safe Rollout Pattern

- keep baseline path available
- gate custom path by feature flag
- collect error/fallback counts and latency deltas

## Promotion Path

When Python shader wrappers are insufficient:

1. port op to extension implementation
2. register with `TORCH_LIBRARY_IMPL(..., MPS, ...)`
3. keep same Python-facing call signature where possible

This minimizes churn for model code using the op.
