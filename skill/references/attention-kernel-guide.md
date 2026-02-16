# Attention Kernel Guide (PyTorch MPS)

## Scope

Use this for custom attention layouts where built-in SDPA is not enough.
Default to `torch.nn.functional.scaled_dot_product_attention` first.

## Rowwise Mapping

Recommended baseline for custom kernels:

1. one threadgroup per `(head, query_position)` row
2. each thread handles strided `d` elements
3. reduce dot-product with simdgroup + threadgroup staging
4. apply online softmax update per key position

This pattern is implemented in `scripts/attention_kernel.py`.

## Online Softmax Update

For each score `s`:

- `new_max = max(row_max, s)`
- `corr = exp(row_max - new_max)`
- `exp_score = exp(s - new_max)`
- `row_sum = row_sum * corr + exp_score`
- `out = out * corr + exp_score * v`

After key loop:

- `out /= row_sum`

## Launch Guidance

- `rows = q_heads * seq`
- `threads = (rows * group_size,)`
- `group_size` should be divisible by `thread_execution_width`
- validate `group_size <= max_threads_per_threadgroup`

## Mask Variants

Common controls:

- `causal`: limit keys to `<= q_pos`
- `window`: limit keys to `[q_pos + 1 - window, ...]`
- `kv_head` mapping for GQA: `kv_head = (q_head * kv_heads) / q_heads`

## Validation Strategy

Always compare against a PyTorch reference:

1. materialize scores with `q @ k^T / sqrt(d)`
2. apply same masks as kernel
3. `softmax(scores) @ v`
4. compare max absolute error on representative shapes

Start with `float32` I/O for kernel validation before exploring lower precision.
