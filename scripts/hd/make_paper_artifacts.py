#!/usr/bin/env python
"""Regenerate CMES paper tables and figures from HD case-study outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_classification_table(case_dir: Path, paper_dir: Path) -> Path | None:
    source = case_dir / "grn" / "classify" / "classify_results.csv"
    if not source.exists():
        return None

    table_dir = paper_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    out = table_dir / "hd_classification_results.csv"
    df.to_csv(out, index=False)
    (table_dir / "hd_classification_results.md").write_text(df.to_markdown(index=False) + "\n")
    return out


def write_auc_increment_figure(case_dir: Path, paper_dir: Path) -> Path | None:
    source = case_dir / "grn" / "classify" / "classify_results.csv"
    if not source.exists():
        return None

    figure_dir = paper_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    labels = df["contrast"].astype(str).tolist()
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar([i - 0.18 for i in x], df["increment_edge"], width=0.36, label="Expression + GRN edge")
    ax.bar(
        [i + 0.18 for i in x], df["increment_coarse"], width=0.36, label="Expression + GRN coarse"
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("AUC increment over expression-only")
    ax.set_title("Topology-dependent HD classification effect")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = figure_dir / "hd_topology_auc_increment.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=repo_root() / "hd_case_study")
    parser.add_argument("--paper-dir", type=Path, default=repo_root() / "paper")
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    case_dir = args.case_dir.resolve()
    outputs = [
        write_classification_table(case_dir, paper_dir),
        write_auc_increment_figure(case_dir, paper_dir),
    ]
    produced = [path for path in outputs if path is not None]
    if not produced:
        print("No HD result files found. Run scripts/hd/run_hd_case_study.py first.")
        return 1
    for path in produced:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
