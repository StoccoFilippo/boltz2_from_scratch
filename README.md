# boltz2_from_scratch

A from-scratch PyTorch reimplementation of Boltz-2.

- **`ROADMAP.md`** — what to build, step by step, with a fake-data testing strategy.
- **`ARCHITECTURE.md`** — how to organize the code, classes, and tests.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Layout

```
src/boltz2/   package source
tests/        tests mirroring src/ one-to-one
configs/      hyperparameter configs (small.yaml for fast fake-data runs)
scripts/      entry points (overfit_one.py is the north-star sanity check)
```
