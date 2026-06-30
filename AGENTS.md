# AGENTS.md — Hunyuan3D Engineering Guide

This file is the operational entry point for coding agents. Humans should
start with `README.md`. Follow the user's latest request first, then this file,
nearer `AGENTS.override.md` files, and the repository's tests and design docs.

## Project truth

- The supported end-to-end path is:
  `hy3d.exe -> PowerShell wrapper -> official Python/CUDA pipeline -> GLB and sidecars`.
- The native GGUF runtime is a tested prototype, not a complete image-to-3D
  backend. Do not describe it as production-complete.
- `todos.md` is the execution ledger, `test.md` defines acceptance, and
  `final.md` records reviewed status. Verify them against the live repository.
- Use the existing CBM project name `cbm+hunyuan` and index the repository
  before CBM-based inspection.

## Repository map

- `src/` — C++17 CLI, backend bridge, GGUF loader, and native primitives.
- `scripts/` — Python pipelines and Windows setup/build/release automation.
- `tests/` — C++, Python, and PowerShell regression tests.
- `benchmarks/` — tensor lookup and real-GGUF loader measurements.
- `tools/` — checkpoint-to-GGUF conversion.
- `examples/` — user-facing generation commands.
- `.github/workflows/` — Windows and Linux CI.

Do not commit `build/`, `dist/`, `outputs/`, `models/`, `.venv*`,
`third_party/`, `.opencode/memory.db*`, checkpoints, GGUF files, or generated
GLBs. Preserve unrelated user changes.

## Working method

1. Explore the relevant code, tests, and `todos.md`/`test.md`/`final.md`.
2. Design the smallest safe change and name its verification gate.
3. For behavior changes, create or update a regression test first.
4. Implement narrowly; avoid opportunistic refactors and new dependencies.
5. Run the narrowest test, then the appropriate broader gates below.
6. Review security boundaries: subprocess arguments, paths, hashes,
   checkpoints, downloads, revisions, secrets, and sidecar error contents.
7. Update docs only when behavior, setup, evidence, or remaining work changes.

Use Team Mode for unfamiliar, cross-cutting, build/release, network,
filesystem, security, or multi-subsystem changes. Keep tiny mechanical edits
single-agent. Subagent findings are advice and must be verified locally.

## Verification gates

Core build and tests:

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
ctest --test-dir build -C Debug -LE slow --output-on-failure --timeout 60
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -LE slow --output-on-failure --timeout 60
```

Release packaging:

```powershell
ctest --test-dir build -C Release -R '^make_release$' --output-on-failure
```

The release test must prove clean configure/build, zip creation, extraction to
a path containing spaces and Unicode, complete `SHA256SUMS.txt` validation,
and executable startup outside the package cwd.

Real CUDA/model checks are slow and require pinned local assets. Run them when
the changed path can affect generation, texture, dependencies, toolchain, or
packaging. Validate GLBs with an independent parser and require successful
JSON sidecars; file existence alone is not acceptance.

The pinned upstream custom-rasterizer patch is applied by
`patch_hy3dpaint_windows.py`; preserve its revision guard and idempotency.
For non-ASCII roots, `build_hy3dpaint_windows.ps1` temporarily maps the common
source/venv root with `subst.exe`, re-enters once through
`-SkipUnicodeRemap`, installs the rasterizer non-editably, and always removes
the mapping in `finally`. Preserve all four parts of this contract.

Setup is intentionally resumable. `hy3d_setup_helpers.ps1` distinguishes
create, reuse, explicit recreate, and incomplete states. Never silently delete
an existing venv; only `-RecreateVenv` may request `uv venv --clear`.

Before completion, parse changed PowerShell/Python files, run
`git diff --check`, inspect `git status`, and report exact commands and
results. Never claim a gate passed unless it completed.

## Safety and release rules

- Never commit secrets or `.env` contents.
- Checkpoint loading remains safe by default; unsafe pickle requires explicit
  `--allow-unsafe-pickle` and trusted input.
- Keep source/model revisions, dependency versions, and binary hashes pinned.
- Reject path traversal, unsafe deletion targets, missing outputs, stale
  outputs, and wrappers that return success without producing their contract.
- Do not force-push, delete user data, change package managers, or add
  production dependencies without explicit authorization.
- Commit and push only when the user asks. Stage explicit intended paths and
  exclude local runtime databases and generated artifacts.
