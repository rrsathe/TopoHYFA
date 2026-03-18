"""
Evaluate HYFA model on 15 target genes: Whole_Blood -> Heart_L_Vent.
Outputs per-gene Pearson/RMSE table, predictions CSV, and ground-truth CSV.
"""

import argparse
import os
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import wandb
from src.data import Data
from src.train_utils import forward
from train_gtex import (
    GTEx_v8_normalised_adata,
    HypergraphDataset,
    HypergraphNeuralNet,
)

GTEX_FILE = "data/GTEX_data.csv"
MODEL_PATH = "data/model.pth"
RESULTS_DIR = "results"

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
args, unknown_args = parser.parse_known_args()

wandb.init(project="multitissue_imputation", config=args.config, mode="disabled")
config: Any = wandb.config

# ── Data ─────────────────────────────────────────────────────────────
print("Loading data...")
adata = GTEx_v8_normalised_adata(file=GTEX_FILE)

# Apply gene-subset filter identical to train_gtex.py
target_genes_df = pd.read_csv("Imputation/output/HYFA_export/target_genes_15.csv", index_col=0)
target_gene_names = target_genes_df.columns.to_numpy(dtype=str)
gene_symbols = np.asarray(adata.var["Symbol"], dtype=str)
gene_mask = np.isin(gene_symbols, target_gene_names)
adata = adata[:, gene_mask].copy()

# Sort columns to match training order
df_var = cast(pd.DataFrame, adata.var).copy()
df_var["orig_idx"] = np.arange(len(df_var))
intersect_genes = [g for g in target_gene_names if g in df_var["Symbol"].values]
df_var = df_var.set_index("Symbol").loc[intersect_genes].reset_index()
adata = adata[:, df_var["orig_idx"].values].copy()

collate_fn = Data.from_datalist

# ── Splits ───────────────────────────────────────────────────────────
train_donors = np.atleast_1d(np.loadtxt("data/splits/gtex_train.txt", delimiter=",", dtype=str))
test_donors = np.atleast_1d(np.loadtxt("data/splits/gtex_test.txt", delimiter=",", dtype=str))
donors = np.asarray(adata.obs["Participant ID"].astype(str).to_numpy(), dtype=str)
train_mask = np.isin(donors, train_donors)
test_mask = np.isin(donors, test_donors)

# ── Model ────────────────────────────────────────────────────────────
device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
config.update(
    {
        "static_node_types": {
            "Tissue": (len(adata.obs["Tissue_idx"].unique()), config.d_tissue),
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

print("Loading model weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ── Evaluation ───────────────────────────────────────────────────────
source_tissues = ["Whole_Blood"]
target_tissues = ["Heart_L_Vent"]

# Build auxiliary dataset for test set
aux_test_dataset = HypergraphDataset(
    adata[test_mask],
    obs_source={"Tissue": source_tissues},
    obs_target={"Tissue": target_tissues},
)
aux_test_loader = DataLoader(
    aux_test_dataset,
    batch_size=len(aux_test_dataset),
    collate_fn=collate_fn,
    shuffle=False,
)

# Also build train auxiliary dataset (needed for TEEBoT baseline later)
aux_train_dataset = HypergraphDataset(
    adata[train_mask],
    obs_source={"Tissue": source_tissues},
    obs_target={"Tissue": target_tissues},
)
aux_train_loader = DataLoader(
    aux_train_dataset,
    batch_size=len(aux_train_dataset),
    collate_fn=collate_fn,
    shuffle=False,
)

print(f"Evaluating: {source_tissues[0]} -> {target_tissues[0]}")

with torch.no_grad():
    d = next(iter(aux_test_loader))
    out, _node_features = forward(d, model, device, preprocess_fn=None)
    y_pred = out["px_rate"].cpu().numpy()
    y_true = d.x_target.cpu().numpy()

gene_symbols = adata.var["Symbol"].values

# Per-gene metrics
pearson_scores = []
rmse_scores = []
for i in range(y_true.shape[1]):
    if np.std(y_true[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
        corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    else:
        corr = 0.0
    pearson_scores.append(corr)
    rmse_scores.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))

mean_pearson = np.nanmean(pearson_scores)
mean_rmse = np.mean(rmse_scores)

print(f"\n--- HYFA Results for {target_tissues[0]} ({len(gene_symbols)} genes) ---")
print(f"Mean Pearson Correlation: {mean_pearson:.4f}")
print(f"Mean RMSE:                {mean_rmse:.4f}\n")

res_df = pd.DataFrame({"Gene": gene_symbols, "Pearson": pearson_scores, "RMSE": rmse_scores})
print(res_df.to_string(index=False))

# ── Save outputs ─────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

res_df.to_csv(f"{RESULTS_DIR}/heart_15_genes_eval.csv", index=False)
print(f"\nSaved per-gene metrics  -> {RESULTS_DIR}/heart_15_genes_eval.csv")

# Participant IDs for rows
participant_ids = [aux_test_dataset.donor_map[p] for p in d.source["Participant ID"].cpu().numpy()]

pred_df = pd.DataFrame(y_pred, columns=gene_symbols, index=participant_ids)
pred_df.index.name = "Participant_ID"
pred_df.to_csv(f"{RESULTS_DIR}/hyfa_predictions_test.csv")
print(f"Saved HYFA predictions  -> {RESULTS_DIR}/hyfa_predictions_test.csv")

truth_df = pd.DataFrame(y_true, columns=gene_symbols, index=participant_ids)
truth_df.index.name = "Participant_ID"
truth_df.to_csv(f"{RESULTS_DIR}/ground_truth_test.csv")
print(f"Saved ground truth      -> {RESULTS_DIR}/ground_truth_test.csv")

# Also save train-set arrays for reuse by benchmark_teebot.py
d_train = next(iter(aux_train_loader))
x_train_source = d_train.x_source.numpy()
y_train_target = d_train.x_target.numpy()
x_train_covs = d_train.source_features["Participant ID"].cpu().numpy()

x_test_source = d.x_source.cpu().numpy()
x_test_covs = d.source_features["Participant ID"].cpu().numpy()

np.savez_compressed(
    f"{RESULTS_DIR}/eval_arrays.npz",
    x_train_source=x_train_source,
    y_train_target=y_train_target,
    x_train_covs=x_train_covs,
    x_test_source=x_test_source,
    x_test_covs=x_test_covs,
    y_test_target=y_true,
    y_test_pred=y_pred,
    gene_symbols=gene_symbols,
)
print(f"Saved reusable arrays   -> {RESULTS_DIR}/eval_arrays.npz")
