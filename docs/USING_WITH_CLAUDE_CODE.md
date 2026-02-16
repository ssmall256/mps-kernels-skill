# Using This Project With Claude Code

## What to Point Claude Code At

Use this repository as your working directory and use `skill/SKILL.md` as the skill entrypoint.

## Suggested Prompting Pattern

- "Use the mps-kernels skill to implement/optimize <operation>."
- "Start from the nearest template in `skill/scripts/` and validate against PyTorch reference ops."
- "Benchmark with `torch.mps.synchronize()` and report speed/correctness tradeoffs."

## Good First Commands

```bash
python -m skill.tests.smoke_test
python skill/scripts/mps_compile_shader_quickstart.py
python skill/scripts/softmax_kernel.py
```

## Tips

- Be explicit about dtype, shape ranges, and tolerance targets.
- Ask for baseline comparisons with `torch.*` implementations.
- Request edge-case tests (small dimensions, odd sizes, non-contiguous inputs).
