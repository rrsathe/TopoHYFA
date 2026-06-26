"""
Trains the model on GTEx data
"""

import argparse
from typing import Any, cast

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import wandb
from torch.utils.data import DataLoader

from src.data import Data
from src.data_utils import load_adjacency_matrix, map_to_ids
from src.dataset import HypergraphDataset
from src.eval_utils import pearson_correlation_score
from src.hnn import HypergraphNeuralNet
from src.train_utils import seed_everything, train

seed_everything(0)
DEFAULT_NUM_WORKERS = 4

GTEX_FILE = "data/GTEX_data.csv"
METADATA_FILE = "data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"


def GTEx(file=GTEX_FILE):
    """
    Loads processed GTEx data
    :param file: path of the CSV file
    :return: Returns
        - data: numpy array of shape=(nb_samples, nb_genes)
        - gene_symbols: numpy array with gene symbols. Shape=(nb_genes,)
        - sampl_ids: numpy array with sample IDs (GTEx IDs of individuals, e.g. GTEX-1117F). Shape=(nb_samples,)
        - tissues: numpy array indicating the tissue of each sample. Shape=(nb_samples,)
    """

    df = pd.read_csv(file, index_col=0)
    tissues = df["tissue"].values
    sampl_ids = df.index.values
    del df["tissue"]
    data = np.float32(df.values)
    gene_symbols = df.columns.values
    return data, gene_symbols, sampl_ids, tissues


def GTEx_metadata(file=METADATA_FILE):
    """
    Loads metadata DataFrame with information about individuals
    :param file: path of the file
    :return: Pandas DataFrame with subjects' information
    """
    df = pd.read_csv(file, delimiter="\t")
    df = df.set_index("SUBJID")
    return df


def GTEx_v8_normalised_adata(file=GTEX_FILE):
    data, gene_symbols, sampl_ids, tissues = GTEx(file=file)
    metadata_df = GTEx_metadata()

    adata = sc.AnnData(data)
    adata.var["Symbol"] = gene_symbols
    adata.obs["Participant ID"] = sampl_ids
    adata.obs["Tissue"] = tissues

    participant_series = adata.obs["Participant ID"]

    adata = adata[participant_series.duplicated(keep=False)]

    tissue_series = adata.obs["Tissue"]
    adata.obs["Tissue_idx"], tissue_dict = map_to_ids(tissue_series.to_numpy())
    adata.uns["Tissue_dict"] = tissue_dict

    participant_series = adata.obs["Participant ID"]
    adata.obs["Participant ID_dyn"] = participant_series

    age_values = metadata_df.loc[participant_series.to_numpy(), "AGE"].to_numpy()
    adata.obs["Age"] = np.asarray([float(a[:2]) for a in age_values], dtype=float)

    sex_values = metadata_df.loc[participant_series.to_numpy(), "SEX"].to_numpy()
    adata.obs["Sex"] = np.asarray(sex_values, dtype=float) - 1

    age_series = adata.obs["Age"]
    sex_series = adata.obs["Sex"]
    donor_age = age_series.to_numpy(dtype=float) / 100
    donor_sex, donor_sex_dict = map_to_ids(sex_series.to_numpy())
    adata.obsm["Participant ID_feat"] = np.stack((donor_age, donor_sex), axis=-1)
    adata.uns["Sex_dict"] = donor_sex_dict

    if adata.X is None:
        raise ValueError("adata.X is None and cannot be stored in layers['x'].")
    adata.layers["x"] = cast(Any, adata.X)

    colors = [
        "#ffaa56",
        "#cdad22",
        "#8fbc8f",
        "#8b1c62",
        "#ee6a50",
        "#ff0000",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#eeee00",
        "#00cdcd",
        "#9ac0cd",
        "#ee82ee",
        "#cdb79e",
        "#eec591",
        "#8b7355",
        "#8b7355",
        "#cdaa7d",
        "#b452cd",
        "#7a378b",
        "#cdb79e",
        "#cdb79e",
        "#9acd32",
        "#cdb79e",
        "#7A67EE",
        "#FFD700",
        "#FFB6C1",
        "#CD9B1D",
        "#B4EEB4",
        "#D9D9D9",
        "#3A5FCD",
        "#1E90FF",
        "#CDB79E",
        "#CDB79E",
        "#FFD39B",
        "#A6A6A6",
        "#008B45",
        "#EED5D2",
        "#EED5D2",
        "#FF00FF",
    ]
    adata.uns["Tissue_colors"] = colors

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="configs/default.yaml", type=str)
    parser.add_argument(
        "--target-genes", type=str, required=True, help="Path to target_genes_15.csv"
    )
    parser.add_argument(
        "--topology-matrix",
        type=str,
        required=True,
        help="Path to adjacency_matrix.csv",
    )
    parser.add_argument("--confounders", type=str, required=True, help="Path to confounders.csv")
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.0,
        help="Graph Laplacian regularization weight",
    )
    parser.add_argument(
        "--source-tissue",
        nargs="+",
        type=str,
        default=["Whole Blood"],
        help="List of source tissue(s)",
    )
    parser.add_argument("--target-tissue", type=str, default="Heart_Atrial", help="Target tissue")
    parser.add_argument(
        "--interpretability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save validation interpretability outputs per batch",
    )
    parser.add_argument(
        "--interpretability-output-dir",
        type=str,
        default="results/interpretability",
        help="Directory for validation interpretability batch files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of DataLoader worker processes",
    )
    args, unknown = parser.parse_known_args()

    run_name = f"{'_'.join(args.source_tissue)}_to_{args.target_tissue}_lambda{args.lambda_reg}"
    wandb.init(
        project="multitissue_imputation",
        entity="r-sathe-indian-institute-of-technology",
        name=run_name,
        config=args.config,
    )
    config: Any = wandb.config
    config.lambda_reg = args.lambda_reg
    print(config)

    adata = GTEx_v8_normalised_adata()

    target_genes_df = pd.read_csv(args.target_genes, index_col=0)
    target_gene_names = target_genes_df.columns.to_numpy()
    symbol_series = cast(pd.Series, adata.var["Symbol"])
    gene_mask = np.isin(symbol_series.to_numpy(), target_gene_names)
    adata = adata[:, gene_mask].copy()

    df_var = adata.var.copy()
    df_var["orig_idx"] = np.arange(len(df_var))
    symbol_series = cast(pd.Series, df_var["Symbol"])
    intersect_genes = [g for g in target_gene_names if g in symbol_series.to_numpy()]
    df_var = df_var.set_index("Symbol").loc[intersect_genes].reset_index()
    adata = adata[:, cast(pd.Series, df_var["orig_idx"]).to_numpy()].copy()

    con_df = pd.read_csv(args.confounders, index_col=0)
    subset_patients = cast(pd.Series, adata.obs["Participant ID"]).to_numpy()
    valid_map = np.isin(subset_patients, con_df.index)

    adjacency_matrix = load_adjacency_matrix(
        args.topology_matrix, cast(pd.Series, adata.var["Symbol"]).to_numpy()
    )

    _, tissue_dict = map_to_ids(adata.obs["Tissue"])
    tissue_dict_inv = {v: k for k, v in tissue_dict.items()}

    donors = cast(pd.Series, adata.obs["Participant ID"]).to_numpy()
    train_donors = np.loadtxt("data/splits/gtex_train.txt", delimiter=",", dtype=str)
    val_donors = np.loadtxt("data/splits/gtex_val.txt", delimiter=",", dtype=str)
    test_donors = np.loadtxt("data/splits/gtex_test.txt", delimiter=",", dtype=str)
    train_mask = np.isin(donors, train_donors)
    test_mask = np.isin(donors, test_donors)
    val_mask = np.isin(donors, val_donors)
    print(len(train_donors), len(val_donors), len(test_donors))

    collate_fn = Data.from_datalist
    dtype = torch.float32
    target_tissues = [args.target_tissue]
    source_tissues = args.source_tissue

    train_dataset = HypergraphDataset(adata[train_mask], dtype=dtype, disjoint=True, static=False)
    val_dataset = HypergraphDataset(
        adata[val_mask],
        dtype=dtype,
        disjoint=False,
        static=True,
        obs_source={"Tissue": source_tissues},
        obs_target={"Tissue": target_tissues},
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    config.static_node_types = {
        "Tissue": (len(adata.obs["Tissue_idx"].unique()), config.d_tissue),
        "metagenes": (config.meta_G, config.d_gene),
    }
    config.dynamic_node_types = {
        "Participant ID": (len(adata.obs["Participant ID"].unique()), config.d_patient)
    }

    config.G = adata.shape[-1]
    model = HypergraphNeuralNet(config).to(device)
    model.metagenes_decoder.adjacency_matrix = adjacency_matrix.to(device)
    model.metagenes_decoder.lambda_reg = args.lambda_reg

    def rho(x, out):
        x_pred = out["px_rate"].detach().cpu().numpy()
        return np.mean(pearson_correlation_score(x, x_pred, sample_corr=True))

    metric_fns = [rho]
    train(
        config,
        model=model,
        loader=train_loader,
        val_loader=val_loader,
        device=device,
        preprocess_fn=None,
        compute_metrics_train=False,
        metric_fns=metric_fns,
        interpretability_output_dir=(
            args.interpretability_output_dir if args.interpretability else None
        ),
    )

    torch.save(model.state_dict(), "data/model.pth")
