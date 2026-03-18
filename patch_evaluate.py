import sys

import nbformat

file_path = "evaluate_GTEx_v8_normalised.ipynb"
try:
    with open(file_path) as f:
        nb = nbformat.read(f, as_version=4)
except Exception as e:
    print(f"Error reading notebook: {e}")
    sys.exit(1)

new_eval_code = """
# --- MODIFIED EVALUATION LOOP ---
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

target_genes_symbols = ['CHDH', 'SREBF1', 'CA14', 'CYP2J2', 'CTF1', 'SNX22', 'ETNPPL', 'SYNPO2L', 'ARHGAP1', 'FAM220A', 'HBA2', 'BLM', 'MAFK', 'HMGN2', 'C4orf46']

# Find gene indices
gene_idx_mask = adata.var['Symbol'].isin(target_genes_symbols)
if 'Symbol' not in adata.var:
    gene_idx_mask = adata.var.index.isin(target_genes_symbols) # fallback
    
target_gene_idxs = np.where(gene_idx_mask)[0]
print(f"Found {len(target_gene_idxs)} target genes out of {len(target_genes_symbols)}")

source_tissues = ['Whole_Blood']
target_tissues = ['Heart_L_Vent']  # Adjusted target tissue as 'Heart' maps to 'Heart_L_Vent' in GTEx context.

print(f"Evaluating Source: {source_tissues[0]} -> Target: {target_tissues[0]}")

aux_test_dataset = HypergraphDataset(adata[test_mask],
                                    obs_source={'Tissue': source_tissues},
                                    obs_target={'Tissue': target_tissues})

aux_test_loader = DataLoader(aux_test_dataset, batch_size=len(aux_test_dataset), collate_fn=collate_fn, shuffle=False)

model.eval()
with torch.no_grad():
    d = next(iter(aux_test_loader))
    out, node_features = forward(d, model, device, preprocess_fn=None)
    y_test_pred = out['px_rate'].cpu().numpy()
    y_test_ = d.x_target.cpu().numpy()

# Isolate the specific targets
y_test_pred_sub = y_test_pred[:, target_gene_idxs]
y_test_sub = y_test_[:, target_gene_idxs]

# Calculate Pearson & RMSE per gene
pearson_scores = []
rmse_scores = []

for i in range(len(target_gene_idxs)):
    # Pearson
    if np.std(y_test_sub[:, i]) > 0 and np.std(y_test_pred_sub[:, i]) > 0:
        corr = np.corrcoef(y_test_sub[:, i], y_test_pred_sub[:, i])[0, 1]
    else:
        corr = 0.0
    pearson_scores.append(corr)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_test_sub[:, i], y_test_pred_sub[:, i]))
    rmse_scores.append(rmse)

mean_pearson = np.nanmean(pearson_scores)
mean_rmse = np.mean(rmse_scores)

print(f"\\n--- Results for target genes in {target_tissues[0]} ---")
print(f"Mean Pearson Correlation: {mean_pearson:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f}\\n")

# Results dataframe
symbols = adata.var['Symbol'].values[target_gene_idxs] if 'Symbol' in adata.var else adata.var.index.values[target_gene_idxs]
res_df = pd.DataFrame({
    'Gene': symbols,
    'Pearson': pearson_scores,
    'RMSE': rmse_scores
})
print("Per-gene details:\\n")
print(res_df.to_string(index=False))
"""

# We search for the baseline loop block that defines `source_tissues = ['Whole_Blood']`
# and runs `for tt in target_tissues:`
found = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and (
        "sample_corr = True\n\ndef rho(x, x_pred):" in cell.source
        or "validate = False\nsource_tissues = ['Whole_Blood']" in cell.source
    ):
        # We replace this massive baseline script cell
        nb.cells[i].source = new_eval_code
        found = True
        print(f"Replaced giant evaluation loop at cell index {i}.")
        break

if not found:
    new_cell = nbformat.v4.new_code_cell(new_eval_code)
    nb.cells.append(new_cell)
    print("Appended the evaluation to the end of notebook.")

with open(file_path, "w") as f:
    nbformat.write(nb, f)
print("Notebook patched fully.")
