# Huntington's Disease Case Study

This directory contains the HD topology-dependent analysis artifact integrated for the
TopoHYFA CMES 2027 submission. The scripts reproduce the route used to test whether
adding a gene-regulatory topology signal improves classification of Huntington's disease
status from TopoHYFA-imputed BA9 frontal cortex expression.

Raw RNA-seq datasets and pretrained checkpoints are intentionally not committed. Place
external inputs under `hd_case_study/prep/` and TopoHYFA/GTEx files under the repository
`data/` directory as described in the root `README.md`.

Run the maintained entry point from the repository root:

```bash
uv run python scripts/hd/run_hd_case_study.py
```

To run only lightweight stages after data preparation:

```bash
uv run python scripts/hd/run_hd_case_study.py --skip-heavy
```

The principal result file is:

```text
hd_case_study/grn/classify/classify_results.csv
```

Regenerate paper tables and figures with:

```bash
uv run python scripts/hd/make_paper_artifacts.py
```
