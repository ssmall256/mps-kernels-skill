# Release Checklist

## Pre-release

- Confirm `README.md` is accurate and up to date
- Update `CHANGELOG.md` for the release version/date
- Confirm `skill/SKILL.md` routing and references are correct
- Review `skill/manifest.txt` for missing/extra files
- Ensure `skill/manifest.txt` is sorted and matches files under `skill/`
- Verify `LICENSE`, `CONTRIBUTING.md`, and `SECURITY.md` are present
- Ensure `.gitignore` covers local/generated artifacts

## Validation

- Run lint:
  - `python -m ruff check skill`
- Run type checks:
  - `python -m mypy skill`
- Run static compile check:
  - `python -m compileall -q skill/kernels skill/scripts skill/tests`
- Run smoke tests:
  - `python -m skill.tests.smoke_test`
  - `MPS_REQUIRED=1 python -m skill.tests.smoke_test` (on Apple Silicon CI/release host)
- Spot-check core scripts:
  - `python skill/scripts/mps_compile_shader_quickstart.py`
  - `python skill/scripts/rmsnorm_kernel.py`
- Build a wheel:
  - `python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/dist .`
- Verify wheel includes skill docs:
  - `skill/SKILL.md`
  - `skill/manifest.txt`
  - `skill/references/*.md`

## Repository Hygiene

- Confirm no secrets or machine-specific paths are committed
- Confirm no accidental binary/cache artifacts are tracked
- Confirm no generated artifacts are tracked (`__pycache__`, `.pyc`, `.DS_Store`)
- Confirm CI workflow is green on the release commit

## GitHub Publishing

- Create/verify repository visibility and description
- Push default branch and tags
- Add release notes (summary + breaking changes + migration notes)
- Verify README rendering and links on GitHub
