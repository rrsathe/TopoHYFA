"""
Visualize top gene contributions from saved HYFA interpretability batches.

Example:
    python visualize_interpretability.py \
        --input-dir results/interpretability \
        --gene-names-csv Imputation/output/HYFA_export/target_genes_15.csv \
        --sample 0 \
        --top-k 15 \
        --output-dir results/interpretability_figures
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top gene contributions from HYFA interpretability outputs"
    )
    parser.add_argument(
        "--input-dir",
        default="results/interpretability",
        type=str,
        help="Directory containing eval_batch_*.pt interpretability files",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Specific batch index to visualize. If omitted, all batch files are used.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample index within each batch. If omitted, all samples are visualized.",
    )
    parser.add_argument("--top-k", type=int, default=15, help="Number of genes to plot")
    parser.add_argument(
        "--score",
        choices=["abs_sum", "signed_sum", "max_abs"],
        default="abs_sum",
        help="How to aggregate metagene contributions into a gene score",
    )
    parser.add_argument(
        "--gene-names-csv",
        type=str,
        default=None,
        help="CSV containing gene names in model order. Uses header columns by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/interpretability_figures",
        type=str,
        help="Directory for PNG figures",
    )
    return parser.parse_args()


def batch_files(input_dir: Path, batch: int | None) -> list[Path]:
    if batch is not None:
        path = input_dir / f"eval_batch_{batch:05d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")
        return [path]

    files = sorted(input_dir.glob("eval_batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No eval_batch_*.pt files found in {input_dir}")
    return files


def load_gene_names(path: str | None, n_genes: int) -> list[str]:
    if path is None:
        return [f"Gene_{i}" for i in range(n_genes)]

    with Path(path).open(newline="") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        raise ValueError(f"Gene name CSV is empty: {path}")

    header = rows[0]
    if len(header) >= n_genes + 1 and header[0] in {"", "Gene", "Participant_ID"}:
        genes = header[1 : n_genes + 1]
    elif len(header) >= n_genes:
        genes = header[:n_genes]
    elif len(rows) >= n_genes:
        genes = [row[0] for row in rows[:n_genes]]
    else:
        raise ValueError(f"Could not read {n_genes} gene names from {path}")

    if len(genes) != n_genes:
        raise ValueError(f"Expected {n_genes} gene names, found {len(genes)} in {path}")
    return genes


def tensor_to_numpy(value: Any) -> np.ndarray:
    import torch

    if not torch.is_tensor(value):
        raise TypeError("Expected a torch tensor in interpretability payload")
    return value.detach().cpu().numpy()


def gene_scores(payload: dict[str, Any], score: str) -> np.ndarray:
    if score == "abs_sum":
        key = "gene_metagene_attribution_abs"
        if key not in payload:
            raise KeyError(f"Missing '{key}' in interpretability payload")
        attribution = tensor_to_numpy(payload[key])
        return attribution.sum(axis=-1)

    key = "gene_metagene_attribution"
    if key not in payload:
        raise KeyError(f"Missing '{key}' in interpretability payload")
    attribution = tensor_to_numpy(payload[key])
    if score == "signed_sum":
        return attribution.sum(axis=-1)
    if score == "max_abs":
        return np.abs(attribution).max(axis=-1)
    raise ValueError(f"Unknown score: {score}")


def plot_sample(
    scores: np.ndarray,
    gene_names: list[str],
    batch_idx: int,
    sample_idx: int,
    top_k: int,
    score_name: str,
    output_dir: Path,
) -> Path:
    sample_scores = scores[sample_idx]
    rank_values = np.abs(sample_scores) if score_name == "signed_sum" else sample_scores
    top_idx = np.argsort(rank_values)[-top_k:]
    top_idx = top_idx[np.argsort(sample_scores[top_idx])]

    values = sample_scores[top_idx]
    labels = [gene_names[i] for i in top_idx]
    colors = np.where(values >= 0, "#2f6f9f", "#b44e4e")

    fig_height = max(4.0, 0.35 * len(labels) + 1.5)
    _, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel(score_name)
    ax.set_ylabel("")
    ax.set_title(f"Top gene contributions: batch {batch_idx}, sample {sample_idx}")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"batch_{batch_idx:05d}_sample_{sample_idx:05d}_{score_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    written: list[Path] = []

    for path in batch_files(input_dir, args.batch):
        import torch

        payload = torch.load(path, map_location="cpu")
        scores = gene_scores(payload, args.score)
        if scores.ndim != 2:
            raise ValueError(f"Expected sample x gene scores, got shape {scores.shape}")

        gene_names = load_gene_names(args.gene_names_csv, scores.shape[1])
        batch_idx = int(path.stem.split("_")[-1])
        sample_indices = [args.sample] if args.sample is not None else list(range(scores.shape[0]))
        for sample_idx in sample_indices:
            if sample_idx < 0 or sample_idx >= scores.shape[0]:
                raise IndexError(
                    f"Sample {sample_idx} is out of range for batch {batch_idx} "
                    f"with {scores.shape[0]} samples"
                )
            written.append(
                plot_sample(
                    scores,
                    gene_names,
                    batch_idx,
                    sample_idx,
                    min(args.top_k, scores.shape[1]),
                    args.score,
                    output_dir,
                )
            )

    print(f"Saved {len(written)} figure(s) to {output_dir}")


if __name__ == "__main__":
    main()
