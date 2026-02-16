# mps-kernels-skill

A practical skill pack for writing **custom Metal compute kernels** from **PyTorch** on Apple Silicon using `torch.mps.compile_shader` and (when needed) C++/Objective-C++ extensions.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U torch
python skill/scripts/mps_compile_shader_quickstart.py
python skill/scripts/rmsnorm_kernel.py
python -m skill.tests.smoke_test
```

If you only need routing and implementation patterns, start with `skill/SKILL.md`.

## Release validation

```bash
python -m pip install --upgrade "ruff>=0.4" "mypy>=1.8"
python -m ruff check skill
python -m mypy skill
python -m compileall -q skill/kernels skill/scripts skill/tests
python -m skill.tests.smoke_test
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/dist .
```

Smoke tests intentionally skip when MPS is unavailable in the runtime.

## Compatibility

- macOS on Apple Silicon (Metal-capable GPU)
- Python 3.10+
- PyTorch 2.3+ with MPS support

## Repository docs

- `skill/SKILL.md`: skill entrypoint and routing guide
- `docs/USING_WITH_CLAUDE_CODE.md`: using this project from Claude Code
- `docs/USING_WITH_CODEX_CHATGPT.md`: using this project from Codex (ChatGPT)
- `CONTRIBUTING.md`: contribution workflow and review expectations
- `SECURITY.md`: vulnerability reporting policy
- `CHANGELOG.md`: release history and notable changes
- `RELEASE_CHECKLIST.md`: pre-publish and release checklist
- `MANIFEST.in`: source distribution include rules

## Structure

- `skill/SKILL.md` - primary skill guide and routing index
- `skill/scripts/` - runnable kernel and benchmark examples
- `skill/kernels/` - reusable helpers (device checks, timing, autotune cache)
- `skill/tests/` - smoke tests for local or CI use
- `skill/references/` - deeper guidance and implementation notes

## Repo Layout

```text
mps-kernels-skill/
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── RELEASE_CHECKLIST.md
├── docs/
│   ├── USING_WITH_CLAUDE_CODE.md
│   └── USING_WITH_CODEX_CHATGPT.md
├── .github/workflows/ci.yml
└── skill/
    ├── SKILL.md
    ├── manifest.txt
    ├── kernels/
    ├── scripts/
    ├── tests/
    └── references/
```
