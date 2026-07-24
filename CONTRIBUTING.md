# Contributing

Thank you for improving TopoHYFA. This repository is maintained as a reproducible
research artifact, so changes should preserve deterministic execution and avoid
committing raw biomedical datasets.

## Development Setup

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run python -m compileall .
```

## Data Policy

Do not commit raw GTEx, SRA, recount3, HD cohort data, checkpoints, or generated
experiment outputs. Use `data/`, `datasets/`, `checkpoints/`, `outputs/`, and
`results/` locally; these paths are ignored.

## Pull Request Expectations

- Keep changes scoped to one reproducible behavior or documentation update.
- Add or update tests for maintained Python modules.
- Document new datasets, seeds, hyperparameters, and output files.
- Record user-facing changes in `CHANGELOG.md`.
- Confirm third-party dependencies are compatible with MIT redistribution.
