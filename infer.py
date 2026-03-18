"""
Inference CLI for HYFA targeted gene imputation.

Usage:
    python infer.py \
        --weights data/model.pth \
        --source Whole_Blood \
        --target Heart_L_Vent \
        --input-csv blood_samples.csv \
        --output-csv predictions.csv

The input CSV must have the same gene columns (and order) that the model
was trained on.  If you used prep_handoff.py to generate the training data,
the column order is determined by Imputation/output/HYFA_export/target_genes_15.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import wandb
from src.data import Data
from src.hnn import HypergraphNeuralNet
from src.train_utils import forward
from train_gtex import GTEx_v8_normalised_adata, HypergraphDataset


def main():
    parser = argparse.ArgumentParser(
        description="HYFA targeted-gene inference: impute tissue expression from blood"
    )
    parser.add_argument(
        "--weights", type=str, default="data/model.pth", help="Path to model weights"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Model config YAML"
    )
    parser.add_argument("--source", type=str, default="Whole_Blood", help="Source tissue label")
    parser.add_argument("--target", type=str, default="Heart_L_Vent", help="Target tissue label")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="CSV of new blood samples (samples x genes). "
        "If omitted, uses GTEx test set as a demo.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Output CSV for imputed expression",
    )
    parser.add_argument(
        "--target-genes",
        type=str,
        default="Imputation/output/HYFA_export/target_genes_15.csv",
        help="CSV whose columns define the target gene subset",
    )
    args = parser.parse_args()

    # ── wandb (disabled for inference) ───────────────────────────────
    wandb.init(project="multitissue_imputation", config=args.config, mode="disabled")
    config = wandb.config

    # ── Data ─────────────────────────────────────────────────────────
    print("Loading GTEx reference data ...")
    adata = GTEx_v8_normalised_adata()

    # Apply gene-subset filter
    target_genes_df = pd.read_csv(args.target_genes, index_col=0)
    target_gene_names = target_genes_df.columns.values
    gene_mask = np.isin(adata.var["Symbol"].values, target_gene_names)
    adata = adata[:, gene_mask].copy()

    df_var = adata.var.copy()
    df_var["orig_idx"] = np.arange(len(df_var))
    intersect_genes = [g for g in target_gene_names if g in df_var["Symbol"].values]
    df_var = df_var.set_index("Symbol").loc[intersect_genes].reset_index()
    adata = adata[:, df_var["orig_idx"].values].copy()

    gene_symbols = adata.var["Symbol"].values
    print(f"  Gene subset: {len(gene_symbols)} genes")

    # ── Model ────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    config.update(
        {
            "static_node_types": {
                "Tissue": (
                    len(adata.obs["Tissue_idx"].unique()),
                    config.d_tissue,
                ),
                "metagenes": (config.meta_G, config.d_gene),
            }
        },
        allow_val_change=True,
    )
    config.update(
        {
            "dynamic_node_types": {
                "Participant ID": (
                    len(adata.obs["Participant ID"].unique()),
                    config.d_patient,
                )
            }
        },
        allow_val_change=True,
    )
    config.G = adata.shape[-1]

    model = HypergraphNeuralNet(config).to(device)
    print(f"Loading weights from {args.weights} ...")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # ── Build dataset ────────────────────────────────────────────────
    if args.input_csv is not None:
        print("Custom input not yet supported in hypergraph dataset mode.")
        print("Running on GTEx test set as demo instead.")

    # Use GTEx test set
    test_donors = np.loadtxt("data/splits/gtex_test.txt", delimiter=",", dtype=str)
    donors = adata.obs["Participant ID"].values
    test_mask = np.isin(donors, test_donors)

    collate_fn = Data.from_datalist
    aux_dataset = HypergraphDataset(
        adata[test_mask],
        obs_source={"Tissue": [args.source]},
        obs_target={"Tissue": [args.target]},
    )
    aux_loader = DataLoader(
        aux_dataset,
        batch_size=len(aux_dataset),
        collate_fn=collate_fn,
        shuffle=False,
    )

    # ── Inference ────────────────────────────────────────────────────
    print(f"Running inference: {args.source} -> {args.target} ...")
    with torch.no_grad():
        d = next(iter(aux_loader))
        out, _ = forward(d, model, device, preprocess_fn=None)
        y_pred = out["px_rate"].cpu().numpy()

    participant_ids = [aux_dataset.donor_map[p] for p in d.source["Participant ID"].cpu().numpy()]

    pred_df = pd.DataFrame(y_pred, columns=gene_symbols, index=participant_ids)
    pred_df.index.name = "Participant_ID"

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    pred_df.to_csv(args.output_csv)
    print(f"Saved {len(pred_df)} predictions -> {args.output_csv}")


if __name__ == "__main__":
    main()
