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

This project is built for **archival scientific reproducibility**. Any researcher can rebuild the exact execution environment and obtain identical results, years into the future.

### Software Environment Specification

- **Host Operating System:** Linux (Ubuntu 22.04 LTS tested) or Windows 10/11 (PowerShell/WSL2)
- **Architecture:** `x86_64` (AMD64)
- **Python Version:** `3.10.16` (pinned explicitly via `requires-python = "==3.10.16"` in `pyproject.toml` and `.python-version`)
- **uv Installer Version:** `0.11.24` (reproducible dependency sync manager)
- **Docker Engine Version:** `20.10+` / `24.0+`
- **Docker Base Image:** `python:3.10.16-slim-bookworm@sha256:f9fd9a142c9e3bc54d906053b756eb7e7e386ee1cf784d82c251cf640c502512`
- **System Locale & Timezone:** `TZ=UTC`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`

### Dependency Pinning Strategy

Every package is locked using `uv.lock`. Major pinned dependencies in `pyproject.toml` include:
- `torch==2.3.0`
- `numpy==1.26.4`
- `scipy==1.13.1`
- `pandas==2.2.2`
- `scikit-learn==1.5.0`
- `statsmodels==0.14.6`
- `anndata==0.11.4`
- `scanpy==1.10.1`
- `biopython==1.87`
- `bioservices==1.11.2`
- `lxml==6.1.1`
- `gseapy==1.2.1`
- `blitzgsea` (pinned to Git commit `de42814395a6cde8404f164b5bff7adbf5df6bbc`)
- `matplotlib==3.9.0`
- `supervenn==0.5.0`
- `umap-learn==0.5.7`
- `h5py==3.16.0`
- `missingpy==0.2.0`
- `nbformat==5.10.4`
- `networkx==3.4.2`
- `pyyaml==6.0.3`
- `tqdm==4.67.3`
- `wandb==0.26.1`

### Rebuilding the Environment from Scratch

To recreate the development virtual environment on your host:

1. **Install `uv`**:
   ```bash
   pip install uv==0.11.24
   ```
2. **Synchronize Dependencies (frozen lockfile)**:
   ```bash
   uv sync --frozen
   ```

### Dataset Integrity Verification

We enforce strict validation of external data inputs. To verify that downloaded files are intact and match expected metadata, run:

```bash
uv run python scripts/verify_datasets.py
```

**Dataset Specifications:**

| Filename | Source | Expected Size (Bytes) | Expected MD5 Hash | Expected SHA256 Hash |
|---|---|---|---|---|
| `GTEX_data.csv.zip` | [Figshare v22650763](https://figshare.com/articles/dataset/Processed_GTEx_v8_data/22650763) | 431,765,777 | `a50db13daf93498136fae21d1302c000` | N/A (MD5 Verified) |
| `GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt` | [GTEx Portal v8](https://gtexportal.org/home/datasets) | 20,271 | `90297fc31512902f4459c757180fe575` | `821bdaff39e7a9a1d166919b3c786724c2b79c2861aeb936a2537a0f59b066f7` |

### Verifying Docker Image Digest

To confirm that your local container matches the audited production image digest, run:

```bash
docker inspect --format='{{index .RepoDigests 0}}' topohyfa:latest
```

### Deterministic ML Execution

Randomness is audited and strictly controlled:
- A unified seeding function `seed_everything` (located in `src/train_utils.py`) fixes random seeds for Python's `random`, `numpy`, and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
- Deterministic PyTorch algorithms are enforced (`torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`, and `torch.use_deterministic_algorithms(True, warn_only=True)`).
- *Nondeterministic Operation Warning:* Note that certain PyTorch operations (like atomic additions during scatter/gather operations in PyTorch Geometric on GPU) can be non-deterministic. In such cases, a warning is printed, but seeds are kept identical.

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

Verify the downloaded datasets' integrity:

```bash
uv run python scripts/verify_datasets.py
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
