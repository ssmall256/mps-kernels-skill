# Contributing

## Scope

This repository contains an agent skill plus runnable PyTorch MPS Metal-kernel examples.
Contributions should improve correctness, clarity, reproducibility, and developer ergonomics.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U torch
pip install -U ruff mypy
python -m ruff check skill
python -m mypy skill
python -m compileall -q skill/kernels skill/scripts skill/tests
python -m skill.tests.smoke_test
```

## Change Guidelines

- Keep examples runnable with minimal setup.
- Prefer small, reviewable pull requests with focused scope.
- Preserve compatibility with Apple Silicon + MPS.
- When adding a new kernel example, include:
  - a validation path against a PyTorch reference implementation where possible
  - at least one benchmark or performance note
  - notes about numeric stability and dtype expectations

## Pull Request Checklist

- Code compiles (`python -m compileall -q skill/kernels skill/scripts skill/tests`)
- Lint passes (`python -m ruff check skill`)
- Type checks pass (`python -m mypy skill`)
- Smoke tests pass or are explicitly skipped with reason
- Docs updated (`README.md`, `skill/SKILL.md`, and/or `skill/references/` as needed)
- No accidental generated artifacts committed (`__pycache__`, `.DS_Store`, etc.)
- Commit messages are descriptive and scoped

## Review Priorities

1. Correctness and regressions
2. Numerical stability and edge cases
3. Runtime behavior on Apple Silicon
4. Documentation clarity
