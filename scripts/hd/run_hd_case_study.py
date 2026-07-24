#!/usr/bin/env python
"""Run the Huntington's disease case-study stages in a reproducible order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    "recoverability": ["02a_gtex_cortex_recoverability.py"],
    "labels": ["02b_blood_labels.py"],
    "grn": ["05_grn_lcc.py"],
    "impute": ["06b_impute_ba9_from_blood.py"],
    "coverage": ["06c_gene_coverage.py"],
    "cortex": ["06a_grn_model_cortex.py", "--imputable-only"],
    "classify": ["06b_classify.py"],
    "diagnostic": ["06d_imputation_diagnostic.py"],
}

DEFAULT_ORDER = [
    "recoverability",
    "labels",
    "grn",
    "impute",
    "coverage",
    "cortex",
    "classify",
    "diagnostic",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stage(case_dir: Path, stage: str, hyfa_root: Path | None) -> None:
    command = [sys.executable, *STAGES[stage]]
    if hyfa_root is not None and stage in {"recoverability", "impute", "diagnostic"}:
        command.extend(["--hyfa-root", str(hyfa_root)])
    print(f"[hd:{stage}] {' '.join(command)}")
    subprocess.run(command, cwd=case_dir, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=repo_root() / "hd_case_study",
        help="Directory containing the HD artifact scripts and generated prep/grn outputs.",
    )
    parser.add_argument(
        "--hyfa-root",
        type=Path,
        default=repo_root(),
        help="TopoHYFA repository root containing train_gtex.py, src/, configs/, and data/.",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        action="append",
        help="Run only the selected stage. Repeat to run multiple stages in order supplied.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip pretrained-HYFA stages that require GTEx data and model checkpoints.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    hyfa_root = args.hyfa_root.resolve() if args.hyfa_root else None
    stages = args.stage or DEFAULT_ORDER
    if args.skip_heavy:
        stages = [s for s in stages if s not in {"recoverability", "impute", "diagnostic"}]

    missing = [
        case_dir / STAGES[stage][0]
        for stage in stages
        if not (case_dir / STAGES[stage][0]).exists()
    ]
    if missing:
        for path in missing:
            print(f"missing stage script: {path}", file=sys.stderr)
        return 2

    for stage in stages:
        run_stage(case_dir, stage, hyfa_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
