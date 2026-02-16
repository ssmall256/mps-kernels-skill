# Attention Variants Guide

## Causal Attention

Rule:

- query at position `i` can only attend to keys `j <= i`

Kernel pattern:

- set `end = min(seq, i + 1)`
- iterate `kv in [start, end)`

## Sliding-Window Attention

Left-window pattern used in this skill:

- set `start = max(0, i + 1 - window)`
- keys `< start` are excluded

Combine with causal by also setting `end = i + 1`.

## Grouped-Query Attention (GQA)

Layout:

- `q`: `(q_heads, seq, d)`
- `k`, `v`: `(kv_heads, seq, d)`
- require `q_heads % kv_heads == 0`

Mapping:

- `kv_head = (q_head * kv_heads) / q_heads`

## Practical Validation Matrix

Validate each variant independently:

1. causal only
2. sliding-window only
3. GQA + causal
4. GQA + causal + window

For each case, compare to reference scores/masks in Python.

## Performance Notes

- Causal and window masks reduce key loop work.
- GQA reduces K/V memory traffic compared with full-head K/V.
- Autotune `group_size` per `(heads, seq, d, mask mode)`.
