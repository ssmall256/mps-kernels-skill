# Using This Project With Codex (ChatGPT)

## Workspace Setup

Open this repository as the active workspace and treat `skill/SKILL.md` as the primary routing guide.

## Suggested Prompting Pattern

- "Use the mps-kernels skill for this task."
- "Pick the closest template in `skill/scripts/` and adapt it for <operation>."
- "Validate numerically against PyTorch reference ops and report max error."
- "Benchmark with synchronized timing and summarize tradeoffs."

## Good First Commands

```bash
python -m skill.tests.smoke_test
python skill/scripts/mps_compile_shader_quickstart.py
python skill/scripts/rmsnorm_kernel.py
python skill/scripts/bench_all.py
```

## Tips

- Ask for concrete file edits and exact validation commands.
- Request release-style summaries: risks, tests run, and what can still fail.
- Keep tasks scoped per PR (one kernel/pattern/benchmark change at a time).
