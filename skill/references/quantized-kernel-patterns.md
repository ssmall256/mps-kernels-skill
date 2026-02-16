# Quantized Kernel Patterns (4-bit)

## Pattern: Packed 4-bit Weights

Store 8 4-bit values per `int32` word:

- packed width: `in_features / 8`
- unpack nibble with:
  - `nibble = (packed >> (4 * nibble_idx)) & 0xF`

## Group-Wise Dequantization

For each group:

- `w = scale[group] * q + bias[group]`

where `q` is the unpacked integer in `[0, 15]`.

## Fused Dequant + Matvec

Hot-path pattern:

1. unpack nibble
2. apply scale/bias
3. multiply by input vector element
4. accumulate and reduce

This avoids materializing a full dequantized matrix on-device.

Reference implementation:

- `scripts/dequant_matvec_kernel.py`

## Shape Constraints

- `group_size % 8 == 0`
- `in_features % group_size == 0`
- `packed.shape == (out_features, in_features / 8)`
- `scales.shape == biases.shape == (out_features, in_features / group_size)`

## Validation

Compare fused kernel output to:

1. explicit unpack + dequant in Python
2. `torch.matmul(dequantized_w, x)`

Use float32 for kernel validation before tuning precision/storage.
