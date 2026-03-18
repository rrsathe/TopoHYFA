"""
TEEBoT baseline benchmark: PCA + Linear Regression on the same
Whole_Blood -> Heart_L_Vent split used by HYFA, producing a
side-by-side comparison bar chart.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.baselines import PCA_linear_regression_baseline

RESULTS_DIR = "results"

# ── Load precomputed arrays from eval_15.py ──────────────────────────
print("Loading cached arrays from eval_15.py ...")
data = np.load(f"{RESULTS_DIR}/eval_arrays.npz", allow_pickle=True)
x_train_source = data["x_train_source"]
y_train_target = data["y_train_target"]
x_train_covs = data["x_train_covs"]
x_test_source = data["x_test_source"]
x_test_covs = data["x_test_covs"]
y_test_target = data["y_test_target"]
y_hyfa_pred = data["y_test_pred"]
gene_symbols = data["gene_symbols"]

print(f"Train samples: {x_train_source.shape[0]}, "
      f"Test samples: {x_test_source.shape[0]}, "
      f"Genes: {len(gene_symbols)}")

# ── TEEBoT baseline ─────────────────────────────────────────────────
# Reshape covariates if multi-tissue (take first tissue slice)
if x_train_covs.ndim == 3:
    x_train_covs = x_train_covs[:, 0, :]
    x_test_covs = x_test_covs[:, 0, :]

print("Running TEEBoT (PCA + Linear Regression) ...")
y_teebot_pred = PCA_linear_regression_baseline(
    x_train_source, y_train_target, x_test_source,
    x_source_covs=x_train_covs,
    x_source_test_covs=x_test_covs,
    n_components=min(30, x_train_source.shape[1] - 1),
)

# ── Metrics ──────────────────────────────────────────────────────────
hyfa_pearson = []
teebot_pearson = []
hyfa_rmse = []
teebot_rmse = []

for i in range(len(gene_symbols)):
    # HYFA
    if np.std(y_test_target[:, i]) > 0 and np.std(y_hyfa_pred[:, i]) > 0:
        hyfa_pearson.append(np.corrcoef(y_test_target[:, i], y_hyfa_pred[:, i])[0, 1])
    else:
        hyfa_pearson.append(0.0)
    hyfa_rmse.append(np.sqrt(mean_squared_error(y_test_target[:, i], y_hyfa_pred[:, i])))

    # TEEBoT
    if np.std(y_test_target[:, i]) > 0 and np.std(y_teebot_pred[:, i]) > 0:
        teebot_pearson.append(np.corrcoef(y_test_target[:, i], y_teebot_pred[:, i])[0, 1])
    else:
        teebot_pearson.append(0.0)
    teebot_rmse.append(np.sqrt(mean_squared_error(y_test_target[:, i], y_teebot_pred[:, i])))

# ── Comparison table ─────────────────────────────────────────────────
comp_df = pd.DataFrame({
    "Gene": gene_symbols,
    "HYFA_Pearson": hyfa_pearson,
    "TEEBoT_Pearson": teebot_pearson,
    "HYFA_RMSE": hyfa_rmse,
    "TEEBoT_RMSE": teebot_rmse,
})

print("\n--- Side-by-side comparison ---")
print(comp_df.to_string(index=False))
print(f"\nHYFA  mean Pearson: {np.nanmean(hyfa_pearson):.4f}  |  TEEBoT mean Pearson: {np.nanmean(teebot_pearson):.4f}")
print(f"HYFA  mean RMSE:    {np.mean(hyfa_rmse):.4f}  |  TEEBoT mean RMSE:    {np.mean(teebot_rmse):.4f}")

comp_df.to_csv(f"{RESULTS_DIR}/hyfa_vs_teebot_comparison.csv", index=False)
print(f"\nSaved comparison CSV -> {RESULTS_DIR}/hyfa_vs_teebot_comparison.csv")

# ── Bar chart ────────────────────────────────────────────────────────
x = np.arange(len(gene_symbols))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Pearson
bars1 = ax1.bar(x - width / 2, hyfa_pearson, width, label="HYFA", color="#4C72B0")
bars2 = ax1.bar(x + width / 2, teebot_pearson, width, label="TEEBoT", color="#DD8452")
ax1.set_ylabel("Pearson Correlation")
ax1.set_title("HYFA vs TEEBoT — Per-Gene Pearson Correlation\n(Whole Blood → Heart Left Ventricle)")
ax1.legend()
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylim(min(min(hyfa_pearson), min(teebot_pearson)) - 0.1, 1.0)

# RMSE
bars3 = ax2.bar(x - width / 2, hyfa_rmse, width, label="HYFA", color="#4C72B0")
bars4 = ax2.bar(x + width / 2, teebot_rmse, width, label="TEEBoT", color="#DD8452")
ax2.set_ylabel("RMSE")
ax2.set_title("HYFA vs TEEBoT — Per-Gene RMSE")
ax2.set_xticks(x)
ax2.set_xticklabels(gene_symbols, rotation=45, ha="right")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/hyfa_vs_teebot_pearson.png", dpi=150, bbox_inches="tight")
print(f"Saved bar chart       -> {RESULTS_DIR}/hyfa_vs_teebot_pearson.png")
