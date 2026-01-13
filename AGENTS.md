# Repository Guidelines

## Project Structure & Module Organization
- `pidsmaker/` holds the core Python package (pipeline stages, models, encoders/decoders, objectives, and utilities).
- `config/` contains YAML system configurations and experiment presets (for example `config/default.yml`, `config/experiments/...`).
- `tests/` hosts functional tests plus `tests/README.md` for running instructions.
- `dataset_preprocessing/` and `postgres/` include dataset setup scripts and DB initialization helpers.
- `docs/` contains the MkDocs site and build scripts; `scripts/` contains run helpers (for example `scripts/run.sh`).

## Build, Test, and Development Commands
- `python pidsmaker/main.py SYSTEM DATASET` runs a local experiment (example: `python pidsmaker/main.py velox CADETS_E3`).
- `./run.sh SYSTEM DATASET` runs in background and logs to `nohup.out`.
- `pytest -v` runs GPU tests; `pytest -v --device cpu -k "not (test_transformations or test_featurizations)"` runs CPU-only tests.
- `pytest --cov=pidsmaker tests/` generates coverage.
- `pre-commit run --all-files` formats and lint-checks before PRs.
- `docs/build.sh` followed by `mkdocs serve --dev-addr=0.0.0.0:8000` builds and serves docs.

## Coding Style & Naming Conventions
- Python: 4-space indentation, 100-character line length (see `pyproject.toml`).
- Ruff is configured for linting and import sorting; keep imports grouped by standard-library, third-party, then `pidsmaker`.
- Use `snake_case` for modules/functions and `CamelCase` for classes; keep config keys in `config/*.yml` consistent with existing patterns.

## Testing Guidelines
- Tests are functional and should be updated when modifying core components.
- Follow existing naming in `tests/` (for example `tests/test_framework.py`).
- Run the targeted CPU/GPU commands above depending on your environment.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative messages (examples: `fix badge`, `update README`, `hotfix: fix ...`).
- Before opening a PR: run `pre-commit`, run relevant tests, and build docs if doc content changed.
- Include a clear description of changes and link relevant issues or papers when applicable.

## Development Tips
- Use the `pids` container for development (see docs) to match runtime dependencies.
- Avoid concurrent runs from scratch with the same dataset/config to prevent file write conflicts; stagger runs or use `--restart_from_scratch`.
