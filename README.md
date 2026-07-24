# Topology-Aware HYFA (TopoHYFA)

TopoHYFA is a topology-aware hypergraph neural network for targeted multi-tissue gene-expression imputation. It extends HYFA with biological topology priors and includes a Huntington's disease (HD) case study prepared for IEEE CMES 2027.

The publication claim demonstrated by the HD extension is:

> Topology-dependent effect demonstrated in Huntington's disease.

## Overview

This repository contains:

- TopoHYFA with graph smoothness regularization
- Standard HYFA and PCA/linear baselines
- GTEx v8 data verification and reproducible training utilities
- Downstream phenotype and disease-status evaluation scripts
- HD BA9 frontal cortex imputation and topology-dependent classification workflow
- IEEE CMES 2027 paper artifact scaffolding

## Scientific Motivation

Many disease-relevant tissues are inaccessible in living subjects. TopoHYFA evaluates whether whole-blood transcriptomes can support targeted imputation of inaccessible tissue expression and whether known biological topology adds information beyond expression values alone.

For HD, the workflow imputes BA9 frontal cortex expression from blood, builds a disease-relevant gene-regulatory network (GRN), and compares HD/control classifiers using:

- expression-only features
- expression plus GRN edge likelihood features
- expression plus coarse topology-community features

An AUC improvement from the topology-aware feature blocks is treated as evidence of a topology-dependent effect.

## Repository Structure

```text
TopoHYFA/
├── configs/
│   └── hd_case_study.yaml
├── hd_case_study/
├── paper/
│   ├── figures/
│   ├── tables/
│   └── supplementary/
├── scripts/
│   ├── hd/
│   └── verify_datasets.py
├── src/
│   └── topohyfa_hd/
├── tests/
├── train_gtex.py
├── student_pipeline.py
├── pyproject.toml
└── uv.lock
```

## Installation

Python `>=3.10,<3.12` is supported. Python 3.10 is the tested environment for the archived lockfile.

```bash
pip install uv==0.11.24
uv sync --frozen
```

For development tools:

```bash
uv sync --group dev
```

## Dataset Preparation

TopoHYFA uses external datasets that are not committed to the repository.

Required GTEx files under `data/`:

```text
data/
├── GTEX_data.csv
├── GTEX_data.csv.zip
├── GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
└── normalised_model_default.pth
```

GTEx download sources:

- Processed GTEx v8 expression: `https://figshare.com/ndownloader/files/40208074`
- GTEx v8 subject phenotypes: `https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt`

Verify downloaded GTEx files:

```bash
uv run python scripts/verify_datasets.py
```

Local-only paths such as `data/`, `datasets/`, `checkpoints/`, `outputs/`, `results/`, `wandb/`, and model checkpoint formats are ignored by git.

## Running Core TopoHYFA

Prepare processed GTEx inputs:

```bash
uv run python prep_handoff.py
```

Run the student-facing TopoHYFA workflow:

```bash
uv run python student_pipeline.py --lambda-reg 0.1
```

Expected core outputs are written under `results/`, including predictions, benchmark metrics, downstream phenotype results, and interpretability figures.

## Huntington's Disease Case Study

The HD artifact is integrated under `hd_case_study/`. It contains the original preparation and analysis scripts plus a maintained runner in `scripts/hd/run_hd_case_study.py`.

Prepare these local HD inputs under `hd_case_study/prep/`:

```text
hd_case_study/prep/
├── hd_blood_counts_by_symbol.csv
├── hd_blood_pheno.csv
├── hd_blood_pheno_full.csv
├── hd_cortex_ba9_counts_by_symbol.csv
└── hd_cortex_ba9_pheno.csv
```

The R scripts in `hd_case_study/` document how the external recount3/SRA-derived inputs were prepared. Do not commit raw cohort data.

Run the complete HD workflow:

```bash
uv run python scripts/hd/run_hd_case_study.py
```

Run selected stages:

```bash
uv run python scripts/hd/run_hd_case_study.py --stage grn --stage coverage --stage classify
```

Regenerate paper tables and figures:

```bash
uv run python scripts/hd/make_paper_artifacts.py
```

Expected HD outputs:

```text
hd_case_study/recoverability/gate_verdict.json
hd_case_study/prep/hd_blood_imputed_ba9.csv
hd_case_study/grn/gene_coverage.csv
hd_case_study/grn/classify/classify_results.csv
hd_case_study/grn/imputation_diagnostic.txt
paper/tables/hd_classification_results.csv
paper/figures/hd_topology_auc_increment.png
```

Reproducibility defaults are recorded in `configs/hd_case_study.yaml`: seed `0`, 5-fold stratified classification, shrinkage `2.0`, source tissue `Whole_Blood`, and target tissue `Brain_Frontal_Cortex`.

## IEEE CMES 2027 Workflow

Conference: IEEE CMES 2027 - International Conference on Medical Engineering and Science
Venue: Indian Institute of Technology Guwahati, Assam, India
Dates: January 21-23, 2027
Paper submission deadline: August 03, 2026
Camera-ready submission: November 23, 2026

Use `paper/` for regenerated figures, tables, and supplementary metadata. Do not place raw datasets or model checkpoints there.

## Quality Checks

Run before release:

```bash
uv run pytest
uv run ruff check .
uv run python -m compileall .
```

## License

TopoHYFA is released under the MIT License. See `LICENSE`.

Third-party datasets remain under their original access terms and licenses. This repository documents external dataset requirements but does not redistribute restricted data.

## Citation

If you use this repository, cite `CITATION.cff` and the submitted CMES 2027 work:

```bibtex
@inproceedings{topohyfa_hd_cmes2027,
  title = {Topology-Dependent Effect Demonstrated in Huntington's Disease with TopoHYFA},
  author = {{TopoHYFA contributors}},
  booktitle = {IEEE CMES 2027 - International Conference on Medical Engineering and Science},
  address = {Indian Institute of Technology Guwahati, Assam, India},
  year = {2027},
  note = {Submitted}
}
```

Please also cite HYFA, TEEBoT, GTEx v8, and the HD source cohorts used in your run.
