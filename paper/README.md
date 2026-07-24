# IEEE CMES 2027 Paper Artifacts

This directory stores reproducible assets for the TopoHYFA Huntington's disease case
study submitted to IEEE CMES 2027.

Expected layout:

```text
paper/
├── figures/
├── tables/
├── supplementary/
└── README.md
```

Regenerate tables and figures from completed HD case-study outputs:

```bash
uv run python scripts/hd/make_paper_artifacts.py
```

The generator reads `hd_case_study/grn/classify/classify_results.csv` and writes:

- `paper/tables/hd_classification_results.csv`
- `paper/tables/hd_classification_results.md`
- `paper/figures/hd_topology_auc_increment.png`

Supplementary material should include protocol notes, command logs, and small derived
metadata only. Do not place raw controlled datasets or model checkpoints in this tree.
