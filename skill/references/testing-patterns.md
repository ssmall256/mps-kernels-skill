# Testing Patterns

## Core Validation Categories

1. Reference parity
2. Shape coverage (small, medium, large)
3. Edge dimensions (`D=1`, odd `D`, large `D`)
4. Dtype coverage (`float32` first, then lower precision)
5. Layout checks (contiguous vs explicitly converted)

## Reference Comparisons

- RMSNorm: compare against `torch.nn.functional.rms_norm`.
- Softmax: compare against `torch.softmax` on same dim.
- Pointwise: compare against canonical op (`torch.nn.functional.silu`, etc).

## Error Metrics

Report at least:

- max absolute error
- (optional) mean absolute error
- row-sum sanity check for probability ops

## Test Harness Tips

- Skip gracefully when MPS is unavailable.
- Seed random generators for reproducibility.
- Keep smoke tests lightweight so CI remains fast.
