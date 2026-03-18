import sys

import nbformat

file_path = "evaluate_GTEx_v8_normalised.ipynb"
try:
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

# We want to add the filtering code right after `adata = GTEx_v8_normalised_adata(file=GTEX_FILE)`
filter_code_to_add = """
import pandas as pd
import numpy as np
# Apply correct bottleneck filtering identical to train_gtex.py
target_genes_df = pd.read_csv("Imputation/output/HYFA_export/target_genes_15.csv", index_col=0)
target_gene_names = target_genes_df.columns.values
gene_mask = np.isin(adata.var["Symbol"].values, target_gene_names)
adata = adata[:, gene_mask].copy()

# Sort columns exactly tracking the target subset order to align adjacency indices
df_var = adata.var.copy()
df_var["orig_idx"] = np.arange(len(df_var))
intersect_genes = [g for g in target_gene_names if g in df_var["Symbol"].values]
df_var = df_var.set_index("Symbol").loc[intersect_genes].reset_index()
adata = adata[:, df_var["orig_idx"].values].copy()
"""

for cell in nb.cells:
    if cell.cell_type == "code":
        if (
            "adata = GTEx_v8_normalised_adata(file=GTEX_FILE)" in cell.source
            and "target_genes_df = pd.read_csv" not in cell.source
        ):
            cell.source = cell.source + "\\n" + filter_code_to_add
            print("Added filtering to adata cell.")
        if (
            "test_mask = np.isin(donors, test_donors)" in cell.source
            and "gtex_test.txt" not in cell.source
        ):
            # Fix split logic
            cell.source = cell.source.replace(
                "train_donors, test_donors = split_patient_train_test(donors, train_rate=0.8)\\n    train_donors, val_donors = split_patient_train_test(train_donors, train_rate=0.75)",
                "test_donors = np.loadtxt('data/splits/gtex_test.txt', delimiter=',', dtype=str)",
            )
            print("Fixed train/test split logic.")

with open(file_path, "w") as f:
    nbformat.write(nb, f)
print("Notebook patched.")
