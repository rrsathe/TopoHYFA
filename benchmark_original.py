import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.data import Data
from src.data_utils import densify, sparsify
from src.dataset import HypergraphDataset
from src.hnn import HypergraphNeuralNet
from train_gtex import GTEx_v8_normalised_adata

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")
console = Console()


class ConfigWrapper:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _unwrap_config_values(obj):
    if isinstance(obj, dict):
        if "value" in obj:
            return _unwrap_config_values(obj["value"])
        return {k: _unwrap_config_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap_config_values(v) for v in obj]
    return obj


def load_config(config_path):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # If it's a Sweep config, values are nested under 'parameters'
    if "parameters" in raw:
        raw = raw["parameters"]

    # Unwrap W&B format: {"key": {"desc": "...", "value": 120}} -> {"key": 120}
    parsed = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            parsed[k] = v["value"]
        else:
            parsed[k] = v

    return ConfigWrapper(**parsed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the original config and data (NO BOTTLENECK)
    log.info("Loading FULL GTEx V8 AnnData object for Original Model...")
    config: Any = load_config("configs/default.yaml")
    adata = GTEx_v8_normalised_adata()

    # Override config to match pre-trained checkpoint dimensions
    config.meta_G = 50

    # Ensure config matches the original massive dimensions
    config.G = adata.shape[-1]
    config.static_node_types = {
        "Tissue": (len(adata.obs["Tissue_idx"].unique()), getattr(config, "d_tissue", 120)),
        "metagenes": (getattr(config, "meta_G", 50), getattr(config, "d_gene", 48)),
    }
    config.dynamic_node_types = {
        "Participant ID": (
            len(adata.obs["Participant ID"].unique()),
            getattr(config, "d_patient", 71),
        )
    }

    # 2. Find the column indices of our 15 target genes in the massive 25k array
    target_genes_df = pd.read_csv("Imputation/output/HYFA_export/target_genes_15.csv", index_col=0)
    target_gene_names = target_genes_df.columns.values

    gene_indices = []
    found_genes = []
    for gene in target_gene_names:
        if gene in adata.var["Symbol"].values:
            idx = np.where(adata.var["Symbol"].values == gene)[0][0]
            gene_indices.append(idx)
            found_genes.append(gene)

    # Convert to numpy array for indexing
    gene_indices = np.array(gene_indices)

    # 3. Setup Test Set (Whole Blood -> Heart)
    donors = adata.obs["Participant ID"].to_numpy(dtype=str)
    test_donors = np.loadtxt("data/splits/gtex_test.txt", delimiter=",", dtype=str)
    test_mask = np.isin(donors, test_donors)

    # Note: Using the exact tissue names as natively found in GTEx
    test_dataset = HypergraphDataset(
        adata[test_mask],
        dtype=torch.float32,
        disjoint=False,
        static=True,
        obs_source={"Tissue": ["Whole_Blood"]},
        obs_target={"Tissue": ["Heart_L_Vent"]},
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, collate_fn=Data.from_datalist, shuffle=False
    )

    # 4. Initialize and Load Original Pre-trained Model
    log.info("Loading original weights from data/normalised_model_default.pth...")
    model = HypergraphNeuralNet(config).to(device)
    model.load_state_dict(torch.load("data/normalised_model_default.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    log.info("Running inference across all 25,000+ genes (This may take a moment)...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Encode source tissue expression to metagene space
            x_source_metagenes = model.encode_metagenes(batch.x_source)

            # Unpack batch into model inputs
            hyperedge_index, hyperedge_attr = sparsify(
                batch.source, model.metagenes, x=x_source_metagenes
            )
            # Move all tensors to device
            hyperedge_index = {k: v.to(device) for k, v in hyperedge_index.items()}
            if hyperedge_attr is not None:
                hyperedge_attr = hyperedge_attr.to(device)

            # Encode: compute node features from source tissue
            node_features = model(
                hyperedge_index, hyperedge_attr, dynamic_node_features=batch.node_features
            )

            # Prepare target hyperedges for prediction
            target_hyperedge_index, _ = sparsify(batch.target, model.metagenes, x=None)
            target_hyperedge_index = {k: v.to(device) for k, v in target_hyperedge_index.items()}

            # Predict: get metagene values for target tissue
            x_pred_metagenes = model.predict(target_hyperedge_index, node_features)

            # Densify predictions back to metagene representation
            x_pred_metagenes_dense = densify(
                batch.target, model.metagenes, target_hyperedge_index, x_pred_metagenes
            )

            # Decode metagenes back to full gene space
            out = model.decode_metagenes(x_pred_metagenes_dense)
            x_pred_genes = out["px_rate"]

            # Extract predictions and targets for 15 genes
            pred = x_pred_genes.cpu().numpy()[:, gene_indices]
            target = batch.x_target.cpu().numpy()[:, gene_indices]

            all_preds.append(pred)
            all_targets.append(target)

    log.info(f"Inference complete. Processed {len(all_preds)} batches.")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if len(all_preds) == 0:
        log.error("No samples were found in the test set. Check tissue filtering parameters.")
        return

    # 5. Calculate Pearson Correlation
    correlations = []
    for i in range(len(found_genes)):
        if np.std(all_targets[:, i]) > 0 and np.std(all_preds[:, i]) > 0:
            corr, _ = pearsonr(all_targets[:, i], all_preds[:, i])
        else:
            corr = 0.0
        correlations.append(corr)

    # 6. Print Results
    table = Table(title="Global HYFA (25k Genes) vs. The 15 Target Markers")
    table.add_column("Gene Symbol", justify="left", style="cyan")
    table.add_column("Original HYFA (r)", justify="right", style="magenta")

    for gene, corr in zip(found_genes, correlations, strict=True):
        table.add_row(gene, f"{corr:.4f}")

    avg_corr = np.mean(correlations)
    table.add_section()
    table.add_row("AVERAGE", f"{avg_corr:.4f}", style="bold white")

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    main()
