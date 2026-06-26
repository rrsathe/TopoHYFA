# Topology-Aware HYFA (TopoHYFA)

Topology-aware hypergraph neural network for targeted multi-tissue gene expression imputation.

This project predicts a targeted cardiovascular gene-expression profile in inaccessible heart tissue using whole-blood transcriptomes from GTEx v8. It extends HYFA with biological topology priors via graph smoothness regularization.

---

## Overview

This repository contains:

- TopoHYFA (Topology-aware HYFA)
- Standard HYFA baseline
- PCA + linear baseline (TEEBoT-inspired)
- Downstream phenotype prediction
- Interpretability and visualization pipeline

The project was developed for academic evaluation of targeted tissue-expression imputation.

---

## Project Goal

Given:
- Whole blood transcriptome data

Predict:
- Left ventricle heart tissue expression
- Focused cardiovascular biomarker panel

The topology-aware model injects biological co-expression structure into training using graph regularization.

---

## Repository Structure

```
TopoHYFA/
├── train_gtex.py
├── eval_15.py
├── benchmark_teebot.py
├── run_disease_prediction.py
├── visualize_interpretability.py
├── prep_handoff.py
├── student_pipeline.py
├── pyproject.toml
├── uv.lock
├── src/
├── data/
├── results/
└── Imputation/
```

---

## Tested Environment

* Python 3.10
* Ubuntu 22.04
* PyTorch 2.x
* CUDA 11.x (optional)

CPU execution works but training is slower.

---

## Installation

This repository uses [`uv`](https://github.com/astral-sh/uv) for reproducible dependency management.

### Install uv

```bash
pip install uv
```

### Clone the repository

```bash
git clone https://github.com/rrsathe/TopoHYFA.git
cd TopoHYFA
```

### Create the environment

```bash
uv sync
```

---

## Reproducibility

This repository targets reproducible, evaluator-friendly execution:

- Uses `uv` for deterministic dependency management and `uv.lock` for exact resolutions.
- Targets CPU-compatible PyTorch wheels for portable execution; GPU is optional if you install CUDA-compatible PyTorch separately.

Reproduce the environment:

```bash
pip install uv
uv sync
```

Run the full pipeline:

```bash
uv run python student_pipeline.py --lambda-reg 0.1
```

---

## Dockerized Execution

We provide a production-grade, containerized setup for reproducible research, testing, and deployment.

### Quick Start with Docker Compose

Ensure you have [Docker](https://www.docker.com/) and Compose installed.

1. **Build the container images**:
   ```bash
   docker compose build
   ```

2. **Run the full imputation pipeline** (automatically mounts local data and results):
   ```bash
   docker compose run --rm app
   ```

### Volume Mounts Configuration

The Docker Compose configuration ([docker-compose.yml](file:///D:/TopoHYFA/docker-compose.yml)) defines three directory mounts:
- **`data/`** (Read-Only): Mounts local GTEx v8 datasets into the container at `/app/data`.
- **`results/`** (Read-Write): Exports model checkpoints, downstream predictions, and interpretability figures back to the host.
- **`configs/`** (Read-Only): Overrides the default model training settings with host configurations.

### Running with Docker CLI (Without Compose)

If you prefer to run using the standard Docker CLI:

```bash
# Build the production image
docker build -t topohyfa:latest .

# Run the student pipeline
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results:rw" \
  -v "$(pwd)/configs:/app/configs:ro" \
  topohyfa:latest python student_pipeline.py --lambda-reg 0.1
```

### Local Development Environment

For active development, running tests, or formatting/linting without local dependency pollution:

```bash
# Start an interactive shell in the development container (with all dev libraries)
docker compose run --rm dev

# Run pytest suite inside the development container
docker compose run --rm dev pytest
```

> [!TIP]
> The development Compose service uses anonymous volumes to insulate the container's `.venv/` from the host's `.venv/`. This prevents compatibility conflicts if you are developing on a Windows host and executing in a Linux container.

---

## Data Preparation

This project uses GTEx v8 data.

### Required Files

Download:

1. GTEx expression matrix
   https://figshare.com/ndownloader/files/40208074

2. GTEx phenotype annotations
   https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt

Place them inside:

```
data/
```

---

### Generate Processed Inputs

Run:

```bash
uv run python prep_handoff.py
```

This creates:

```
Imputation/output/HYFA_export/
├── target_genes_15.csv
├── adjacency_matrix.csv
└── confounders.csv
```

---

## Quick Start (Recommended)

Run the complete student-facing pipeline:

```bash
uv run python student_pipeline.py --lambda-reg 0.1
```

This automatically:

1. Trains TopoHYFA
2. Evaluates predictions
3. Benchmarks against PCA baseline
4. Runs downstream SEX prediction
5. Generates interpretability visualizations

---

## Reusing Existing Trained Models

Skip training and reuse `data/model.pth`:

```bash
uv run python student_pipeline.py --skip-train
```

---

## Pipeline Outputs

After execution, the following outputs are generated:

```
results/
├── interpretability/
├── interpretability_figures/
├── hyfa_predictions_test.csv
├── benchmark_results.csv
└── disease_prediction_results.csv
```

---

## Expected Runtime

Approximate runtime:

| Hardware | Runtime       |
| -------- | ------------- |
| CPU      | 1–3 hours     |
| GPU      | 10–30 minutes |

Depends on:

* number of epochs
* hardware
* GTEx preprocessing

---

## Example Workflow

```bash
# Step 1
uv run python prep_handoff.py

# Step 2
uv run python student_pipeline.py --lambda-reg 0.1
```

---

## Core Components

**TopoHYFA** — Topology-aware hypergraph factorization model with graph smoothness regularization.

**HYFA Baseline** — Original hypergraph factorization framework without topology regularization.

**TEEBoT Baseline** — PCA + linear regression baseline inspired by tissue-expression estimation methods.

**Downstream Evaluation** — SEX phenotype prediction using imputed expression profiles.

**Interpretability** — Feature attribution and topology visualization tools.

---

## Results Summary

| Model | Avg Pearson r | Notes |
|---|---|---|
| TEEBoT | 0.369 | PCA + LASSO baseline |
| Global HYFA | 0.424 | Best global average |
| Topology-Aware HYFA | 0.310 | Improved targeted biomarkers |

Key improvements observed on:

* CYP2J2
* biologically co-expressed cardiovascular markers

---

## Interpretability Outputs

The repository generates:

* gene attribution plots
* topology-aware importance visualizations
* prediction comparison figures

Generated figures are saved in:

```
results/interpretability_figures/
```

---

## Common Issues

### Missing GTEx files

Ensure required files exist in:

```
data/
```

### CUDA issues

The pipeline supports CPU execution if CUDA is unavailable.

### Matplotlib cache errors

The pipeline automatically sets:

```python
MPLCONFIGDIR=/tmp/matplotlib
```

---

## Development

We establish a robust quality assurance and static analysis toolchain to ensure code correctness, safety, and compatibility.

### Installation & Setup

Install the project along with its development and quality assurance dependencies:

```bash
uv sync --extra dev
```

This installs the standard scientific dependencies plus code checkers, type analysers, security scanners, and test coverage tools.

### Automated Pre-commit Hooks

We use `pre-commit` to execute checks automatically before each commit. The configuration is stored in [.pre-commit-config.yaml](file:///D:/TopoHYFA/.pre-commit-config.yaml) and executes:
- Trailing whitespace removal
- EOF fixing
- Ruff lint auto-fixes and formatting
- Static type checking with `ty check` (mypy is intentionally omitted to standardize type checking)
- Dependency auditing with `deptry`

Register the pre-commit hooks:
```bash
uv run pre-commit install
```

### Unified QA and Task Runner

A unified, cross-platform runner script ([scripts/qa.py](file:///D:/TopoHYFA/scripts/qa.py)) is provided to execute development actions. Run them via `uv run python scripts/qa.py <action>`:

* **Formatting**: `uv run python scripts/qa.py format` (formats Python code via Ruff)
* **Linting**: `uv run python scripts/qa.py lint` (lints code; use `--fix` to auto-fix safe errors)
* **Type Checking**: `uv run python scripts/qa.py typecheck` (runs `ty check` for static type checking; mypy is not used)
* **Testing**: `uv run python scripts/qa.py test` (runs unit tests via pytest)
* **Coverage**: `uv run python scripts/qa.py cov` (runs pytest and generates detailed coverage data)
* **Security**: `uv run python scripts/qa.py security` (audits packages via `pip-audit`)
* **Dependencies**: `uv run python scripts/qa.py deptry` (analyzes imports and usage via `deptry`)
* **Docker Build**: `uv run python scripts/qa.py docker-build` (builds the production Docker image)
* **Docker Test**: `uv run python scripts/qa.py docker-test` (verifies Compose config and performs package import smoke test)

### Running the Full QA Pipeline

To run the complete quality assurance suite recursively:
```bash
uv run python scripts/qa.py qa
```

### Interpreting Test Coverage

When running the coverage tool (`cov`), Pytest executes the suite under `pytest-cov`, tracking branch execution paths.
* **Terminal Summary**: Pytest prints a grid showing coverage percentage per file, indicating exactly which line ranges (in the `Missing` column) were skipped.
* **HTML Report**: Detailed visualization is compiled to the `htmlcov/` directory. Open `htmlcov/index.html` in your browser to interactively trace executed and missing lines.

---

## Citation

If you use this repository, please cite HYFA and TEEBoT (see original repo citations).

---

## Acknowledgements

This project builds upon:

* HYFA
* TEEBoT
* GTEx v8
