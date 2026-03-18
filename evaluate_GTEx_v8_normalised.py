#!/usr/bin/env python

# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import argparse
from collections import Counter
from typing import Any, cast

import blitzgsea as blitz
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from Bio.KEGG import REST
from matplotlib.ticker import MultipleLocator
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.baselines import PCA_linear_regression_baseline, impute_knn
from src.data import Data
from src.data_utils import split_patient_train_test
from src.dataset import HypergraphDataset
from src.eval_utils import pearson_correlation_score
from src.hnn import HypergraphNeuralNet
from src.train_utils import forward
from train_gtex import GTEx_v8_normalised_adata

sns.set_style("whitegrid")


# In[ ]:


RESULTS_DIR = "results"
MODEL_PATH = "data/normalised_model_default.pth"
GTEX_FILE = "data/GTEX_data.csv"
METADATA_FILE = "data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--config", dest="config", default="configs/default.yaml", type=str)
args, unknown = parser.parse_known_args()

# Initialise wandb
wandb.init(project="multitissue_imputation", config=args.config, mode="disabled")
config: Any = wandb.config
print(config)


# In[ ]:


# Load data
adata = GTEx_v8_normalised_adata(file=GTEX_FILE)
tissue_dict = adata.uns["Tissue_dict"]
tissue_dict_inv = {v: k for k, v in tissue_dict.items()}


# In[ ]:


collate_fn = Data.from_datalist

# Split train/val/test
donors = adata.obs["Participant ID"].values
train_donors, test_donors = split_patient_train_test(donors, train_rate=0.8)
train_donors, val_donors = split_patient_train_test(train_donors, train_rate=0.75)
train_mask = np.isin(donors, train_donors)
test_mask = np.isin(donors, test_donors)
val_mask = np.isin(donors, val_donors)

train_dataset = HypergraphDataset(adata[train_mask], disjoint=True, static=False)
val_dataset = HypergraphDataset(adata[val_mask], disjoint=True, static=True)
test_dataset = HypergraphDataset(adata[test_mask], static=True)
train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False
)
test_loader = DataLoader(
    test_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=False
)


# In[ ]:


# Use certain GPU
device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

# Select dynamic/static node types
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
            "Participant ID": (len(adata.obs["Participant ID"].unique()), config.d_patient)
        }
    },
    allow_val_change=True,
)

# Model
config.G = adata.shape[-1]
model = HypergraphNeuralNet(config).to(device)  # .double()


# In[ ]:


model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
# #### Plot tissue embeddings

# In[ ]:


params = model.params["Tissue"].cpu().detach().numpy()

tissue_params_2d = TSNE(
    n_components=2, learning_rate="auto", init="random", random_state=0
).fit_transform(params)

plt.figure(figsize=(8, 8))
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
ax = plt.gca()
x1, x2 = tissue_params_2d.T
ax.scatter(x1, x2, c=colors, s=300)

for t, i in tissue_dict.items():
    x_coord = x1[i]
    y_coord = x2[i]
    txt = t.replace("_", " ").replace("Brain", "")

    ax.annotate(
        txt,
        (x_coord, y_coord),
        textcoords="offset points",  # how to position the text
        xytext=(0, 10),  # distance from text to points (x,y)
        fontsize=12,
        # fontweight='bold',
        # rotation=45,
        ha="center",
    )
plt.axis("off")
# plt.title('Tissue embeddings from multi-tissue imputation model', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/tsne_tissue_embeddings.pdf", bbox_inches="tight")
# In[ ]:


params = model.params["Tissue"].cpu().detach().numpy()

tissue_params_2d = umap.UMAP().fit_transform(params)

plt.figure(figsize=(8, 8))
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
ax = plt.gca()
x1, x2 = tissue_params_2d.T
ax.scatter(x1, x2, c=colors, s=300)

for t, i in tissue_dict.items():
    x_coord = x1[i]
    y_coord = x2[i]
    txt = t.replace("_", " ").replace("Brain", "")

    ax.annotate(
        txt,
        (x_coord, y_coord),
        textcoords="offset points",  # how to position the text
        xytext=(0, 10),  # distance from text to points (x,y)
        fontsize=12,
        # fontweight='bold',
        # rotation=45,
        ha="center",
    )


plt.axis("off")
# plt.title('Tissue embeddings from multi-tissue imputation model', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/figures/umap_tissue_embeddings.pdf", bbox_inches="tight")
# #### Example: predictions validation set

# In[ ]:


d = next(iter(val_loader))


# In[ ]:


with torch.no_grad():
    out, node_features = forward(d, model, device, preprocess_fn=None, use_latent_mean=True)
    x_pred = torch.distributions.normal.Normal(loc=out["px_rate"], scale=out["px_r"]).mean


# In[ ]:


idx = 9228
plt.scatter(d.x_target[:, idx].cpu().numpy(), x_pred[:, idx].cpu().numpy())


# #### Compare to baselines

# In[ ]:


# --- MODIFIED EVALUATION LOOP ---
target_genes_symbols = [
    "CHDH",
    "SREBF1",
    "CA14",
    "CYP2J2",
    "CTF1",
    "SNX22",
    "ETNPPL",
    "SYNPO2L",
    "ARHGAP1",
    "FAM220A",
    "HBA2",
    "BLM",
    "MAFK",
    "HMGN2",
    "C4orf46",
]

# Find gene indices
gene_idx_mask = adata.var["Symbol"].isin(target_genes_symbols)
if "Symbol" not in adata.var:
    gene_idx_mask = adata.var.index.isin(target_genes_symbols)  # fallback

target_gene_idxs = np.where(gene_idx_mask)[0]
print(f"Found {len(target_gene_idxs)} target genes out of {len(target_genes_symbols)}")

source_tissues = ["Whole_Blood"]
target_tissues = [
    "Heart_L_Vent"
]  # Adjusted target tissue as 'Heart' maps to 'Heart_L_Vent' in GTEx context.

print(f"Evaluating Source: {source_tissues[0]} -> Target: {target_tissues[0]}")

aux_test_dataset = HypergraphDataset(
    adata[test_mask], obs_source={"Tissue": source_tissues}, obs_target={"Tissue": target_tissues}
)

aux_test_loader = DataLoader(
    aux_test_dataset, batch_size=len(aux_test_dataset), collate_fn=collate_fn, shuffle=False
)

model.eval()
with torch.no_grad():
    d = next(iter(aux_test_loader))
    out, node_features = forward(d, model, device, preprocess_fn=None)
    y_test_pred = out["px_rate"].cpu().numpy()
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

print(f"\n--- Results for target genes in {target_tissues[0]} ---")
print(f"Mean Pearson Correlation: {mean_pearson:.4f}")
print(f"Mean RMSE: {mean_rmse:.4f}\n")

# Results dataframe
symbols = (
    adata.var["Symbol"].values[target_gene_idxs]
    if "Symbol" in adata.var
    else adata.var.index.values[target_gene_idxs]
)
res_df = pd.DataFrame({"Gene": symbols, "Pearson": pearson_scores, "RMSE": rmse_scores})
print("Per-gene details:\n")
print(res_df.to_string(index=False))
results_df = pd.DataFrame(columns=["score", "source", "target", "method"])


# In[ ]:


sns.set(font_scale=1.6)
plt.figure(figsize=(20, 3))
sns.barplot(
    y="score",
    x="target",
    hue="method",
    data=results_df[(results_df["method"] == "HYFA (blood)") | (results_df["method"] == "TEEBoT")],
    order=np.unique(results_df["target"]),
)  # capsize = 0.1
plt.xticks(rotation=45, ha="right")

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
plt.title("Prediction performance with whole blood as source")
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=2)
# plt.savefig(f'{RESULTS_DIR}/comparison_scores_pertissue_blood_sample_corr{sample_corr}.pdf', bbox_inches='tight')


# In[ ]:


sns.reset_orig()
sns.set_style("whitegrid")

results_df["score"] = pd.to_numeric(results_df["score"])
ranks = results_df.groupby("method")["score"].median().fillna(0).sort_values().index

sns.boxplot(x="method", y="score", data=results_df, order=ranks)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.title("Performance with whole blood as source")
# plt.savefig(f'{RESULTS_DIR}/aggregated_scores_blood_sample_corr{sample_corr}.pdf', bbox_inches='tight');


# In[ ]:


sns.reset_orig()
sns.set_style("whitegrid")

results_df["score"] = pd.to_numeric(results_df["score"])
ranks = results_df.groupby("method")["score"].median().fillna(0).sort_values().index

sns.boxplot(x="method", y="score", data=results_df, order=ranks)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.title("Performance with whole blood as source")
# plt.savefig(f'{RESULTS_DIR}/aggregated_scores_blood_sample_corr{sample_corr}.pdf', bbox_inches='tight');


# #### Comparison with TEEBoT across multiple tissues

# In[ ]:


# Pre-load all data for kNN imputation
train_loader_all = DataLoader(
    train_dataset, batch_size=len(train_dataset), collate_fn=collate_fn, shuffle=False
)
d = next(iter(train_loader_all))
y_observed = np.full((len(train_dataset), len(tissue_dict), config.G), np.nan)
y_observed[d.source["Participant ID"].numpy(), d.source["Tissue"].numpy(), :] = d.x_source
y_covs = d.node_features["Participant ID"].cpu().numpy()


# In[ ]:


sample_corr = True
score_fn = pearson_correlation_score

source_tissues = ["Whole_Blood", "Skin_Sun_Epsd", "Skin_Not_Sun_Epsd", "Adipose_Subcutaneous"]
ttissues = list(tissue_dict.keys())
ttissues = [t for t in ttissues if t not in source_tissues]
source_name = "Accessible tissues"

results_df = pd.DataFrame([], columns=["score", "source", "target", "method"])

for t in ttissues:
    print(", ".join(source_tissues), "to", t)
    target_tissues = [t]
    target_name = t.replace("_", " ")

    # Train set
    split_mask = train_mask
    valid_donors = []
    donors = adata[split_mask].obs["Participant ID"].unique()
    for donor in donors:
        donor_mask = adata[split_mask].obs["Participant ID"] == donor
        all_tissues_collected = all(
            t in adata[split_mask].obs[donor_mask]["Tissue"].values
            for t in source_tissues + target_tissues
        )
        if all_tissues_collected:
            valid_donors.append(donor)
    aux_train_dataset = HypergraphDataset(
        adata[split_mask],
        obs_source={"Tissue": source_tissues, "Participant ID": valid_donors},
        obs_target={"Tissue": target_tissues, "Participant ID": valid_donors},
        static=True,
        verbose=True,
    )
    aux_train_loader = DataLoader(
        aux_train_dataset, batch_size=len(aux_train_dataset), collate_fn=collate_fn, shuffle=False
    )

    # Eval set
    split_mask = test_mask
    valid_donors = []
    donors = adata[split_mask].obs["Participant ID"].unique()
    for donor in donors:
        donor_mask = adata[split_mask].obs["Participant ID"] == donor
        all_tissues_collected = all(
            t in adata[split_mask].obs[donor_mask]["Tissue"].values
            for t in source_tissues + target_tissues
        )
        if all_tissues_collected:
            valid_donors.append(donor)

    aux_val_dataset = HypergraphDataset(
        adata[split_mask],
        obs_source={"Tissue": source_tissues, "Participant ID": valid_donors},
        obs_target={"Tissue": target_tissues, "Participant ID": valid_donors},
        static=True,
        verbose=True,
    )
    aux_val_loader = DataLoader(
        aux_val_dataset, batch_size=len(aux_val_dataset), collate_fn=collate_fn, shuffle=False
    )

    it = iter(aux_val_loader)
    val_d = next(it)
    patients_source_val = val_d.source["Participant ID"].cpu().numpy()

    print(source_tissues, target_name, len(np.unique(patients_source_val)))
    if len(np.unique(patients_source_val)) >= 25:  # combinations with >= 25 patients
        # Reshape and concatenate multiple tissues
        it = iter(aux_train_loader)
        d = next(it)
        x_source = d.x_source.reshape(-1, len(source_tissues) * d.x_source.shape[-1])
        x_target = d.x_target  # .reshape(-1, len(source_tissues)* d.x_source.shape[-1])
        x_source_val = val_d.x_source.reshape(-1, len(source_tissues) * val_d.x_source.shape[-1])
        x_target_val = (
            val_d.x_target
        )  # .reshape(-1, len(source_tissues) * val_d.x_source_val.shape[-1])
        x_source_covs = d.source_features["Participant ID"].cpu().numpy()
        x_source_val_covs = val_d.source_features["Participant ID"].cpu().numpy()
        x_source_covs = x_source_covs.reshape(-1, len(source_tissues), x_source_covs.shape[-1])[
            :, 0, :
        ]
        x_source_val_covs = x_source_val_covs.reshape(
            -1, len(source_tissues), x_source_val_covs.shape[-1]
        )[:, 0, :]

        # Blood surrogate baseline
        blood_source_mask = val_d.source["Tissue"] == 48
        scores = score_fn(
            x_target_val.numpy(), val_d.x_source[blood_source_mask].numpy(), sample_corr=sample_corr
        )
        print(f"Blood surrogate baseline: \n Mean score: {scores.mean()}")

        # Append results
        df_ = pd.DataFrame(
            {
                "score": scores,
                "source": [source_name] * len(scores),
                "target": [target_name] * len(scores),
                "method": ["Blood surrogate"] * len(scores),
            }
        )
        results_df = pd.concat([results_df, df_])

        # Mean baseline
        if sample_corr:
            means = d.x_target.mean(axis=0).numpy()
            y_test_pred = np.repeat(means[None, :], x_target_val.shape[0], axis=0)
            scores = score_fn(x_target_val.numpy(), y_test_pred, sample_corr=sample_corr)
            print(f"Mean baseline: \n Mean score: {scores.mean()}")

            # Append results
            df_ = pd.DataFrame(
                {
                    "score": scores,
                    "source": [source_name] * len(scores),
                    "target": [target_name] * len(scores),
                    "method": ["mean"] * len(scores),
                }
            )
            results_df = pd.concat([results_df, df_])

        # KNN baseline
        x_train_knn = np.concatenate((x_source, x_target), axis=-1)
        test_nans = np.full((x_source_val.shape[0], x_target.shape[1]), np.nan)
        x_test_knn = np.concatenate((x_source_val, test_nans), axis=-1)
        x_knn = np.concatenate((x_train_knn, x_test_knn), axis=0)

        x_knn_covs = np.concatenate((x_source_covs, x_source_val_covs), axis=0)
        knn_imp = impute_knn(x_knn, covariates=x_knn_covs, k=20)
        knn_imp_ = knn_imp[x_source.shape[0] :, x_source.shape[1] :]
        scores = score_fn(x_target_val.numpy(), knn_imp_, sample_corr=sample_corr)
        print(f"kNN baseline: \n Mean score: {scores.mean()}")

        # Append results
        df_ = pd.DataFrame(
            {
                "score": scores,
                "source": [source_name] * len(scores),
                "target": [target_name] * len(scores),
                "method": ["kNN"] * len(scores),
            }
        )
        results_df = pd.concat([results_df, df_])

        # TEEBoT
        x_target_pred = PCA_linear_regression_baseline(
            x_source.numpy(),
            x_target.numpy(),
            x_source_val.numpy(),
            x_source_covs=x_source_covs,
            x_source_test_covs=x_source_val_covs,
        )
        scores = score_fn(x_target_val.numpy(), x_target_pred, sample_corr=sample_corr)
        print(f"TEEBoT regression baseline: \n Mean score: {scores.mean()}")

        # Store results
        df_ = pd.DataFrame(
            {
                "score": scores,
                "source": [source_name] * len(scores),
                "target": [target_name] * len(scores),
                "method": ["TEEBoT"] * len(scores),
            }
        )
        results_df = pd.concat([results_df, df_])

        # Hypergraph
        with torch.no_grad():
            d = next(iter(aux_val_loader))
            out, node_features = forward(d, model, device, preprocess_fn=None)
            y_pred = out["px_rate"].cpu().numpy()
            y_ = d.x_target.cpu().numpy()
        assert np.allclose(y_, x_target_val)

        scores = score_fn(x_target_val.numpy(), y_pred, sample_corr=sample_corr)
        print(f"Hypergraph neural network (accessible): \n Mean score: {scores.mean()}")

        # Store results
        df_ = pd.DataFrame(
            {
                "score": scores,
                "source": [source_name] * len(scores),
                "target": [target_name] * len(scores),
                "method": ["HYFA (accessible)"] * len(scores),
            }
        )
        results_df = pd.concat([results_df, df_])

        # Hypergraph baseline (all tissues)
        # Select same set of individuals
        aux_val_dataset_ = HypergraphDataset(
            adata[split_mask],
            obs_source={
                "Participant ID": list(aux_val_dataset.donor_map.values()),
                "Tissue": [k for k in adata.uns["Tissue_dict"] if k != t],
            },
            obs_target={"Tissue": [t]},
            static=True,
        )
        aux_val_loader_ = DataLoader(
            aux_val_dataset_, batch_size=len(aux_val_dataset_), collate_fn=collate_fn, shuffle=False
        )

        # Compute predictions and score
        model.eval()
        with torch.no_grad():
            d = next(iter(aux_val_loader_))

            out, node_features = forward(d, model, device, preprocess_fn=None)
            y_val_pred = (
                out["px_rate"].cpu().numpy()
            )  # torch.distributions.normal.Normal(loc=out['px_rate'], scale=out['px_r']).mean.cpu().numpy()
            y_test_ = d.x_target.cpu().numpy()

        scores = score_fn(x_target_val.numpy(), y_val_pred, sample_corr=sample_corr)
        print(f"Hypergraph neural network (all): \n Mean score: {scores.mean()}")

        # Append results
        df_ = pd.DataFrame(
            {
                "score": scores,
                "source": [source_name] * len(scores),
                "target": [target_name] * len(scores),
                "method": ["HYFA (all)"] * len(scores),
            }
        )
        results_df = pd.concat([results_df, df_])


# In[ ]:


sns.set(font_scale=1.6)
plt.figure(figsize=(20, 4))
sns.barplot(
    y="score",
    x="target",
    hue="method",
    data=results_df[
        (results_df["method"] == "HYFA (accessible)") | (results_df["method"] == "TEEBoT")
    ],
    order=np.unique(results_df["target"]),
)
plt.xticks(rotation=45, ha="right")

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0)
plt.title(
    "Prediction performance with accessible tissues as source tissues (whole blood, skin, and adipose subcutaneous)"
)
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.75), fancybox=True, shadow=True, ncol=2)
# plt.savefig(f'{RESULTS_DIR}/comparison_accessible_scores_valtest_sample_corr{sample_corr}.pdf', bbox_inches='tight')


# In[ ]:


sns.reset_orig()
sns.set_style("whitegrid")

results_df["score"] = pd.to_numeric(results_df["score"])
ranks = results_df.groupby("method")["score"].median().fillna(0).sort_values().index

sns.boxplot(x="method", y="score", data=results_df, order=ranks)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.title("Performance with accessible tissues as source")
# plt.ylim((-0.75, 0.85))
# plt.savefig(f'{RESULTS_DIR}/aggregated_scores_test_accessible_sample_corr{sample_corr}.pdf', bbox_inches='tight');


# In[ ]:


results_df.to_csv(f"{RESULTS_DIR}/results_pertissue_test_sources_accessible.csv")


# In[ ]:


baseline_means = results_df[results_df["method"] == "TEEBoT"].groupby("target")["score"].mean()
ours_means = (
    results_df[results_df["method"] == "HYFA (accessible)"].groupby("target")["score"].mean()
)
(ours_means > baseline_means).sum(), ours_means.size


# #### Increase in performance when multiple accessible tissues are used as source

# In[ ]:


score_fn = pearson_correlation_score

source_tissues = ["Whole_Blood", "Skin_Sun_Epsd", "Skin_Not_Sun_Epsd", "Adipose_Subcutaneous"]
source_tissues_idxs = [tissue_dict[t] for t in source_tissues]
names = [t.replace("_", " ") for t in source_tissues] + ["Accessible All"]

scores_col = []
source_col = []
target_col = []

# Target tissue
ttissues = list(tissue_dict.keys())  # - set(['Testis', 'Cells_Cultured'])
ttissues = [t for t in ttissues if t not in source_tissues]

for t in ttissues:
    target_tissues = [t]
    target_name = t.replace("_", " ")

    # Get data
    # split_mask = val_mask
    split_mask = np.logical_or(val_mask, test_mask)

    donors = adata[split_mask].obs["Participant ID"].unique()

    valid_donors = []
    for donor in donors:
        donor_mask = adata[split_mask].obs["Participant ID"] == donor
        all_tissues_collected = all(
            t in adata[split_mask].obs[donor_mask]["Tissue"].values
            for t in source_tissues + target_tissues
        )
        if all_tissues_collected:
            valid_donors.append(donor)

    aux_dataset = HypergraphDataset(
        adata[split_mask],
        obs_source={"Tissue": source_tissues, "Participant ID": valid_donors},
        obs_target={"Tissue": target_tissues, "Participant ID": valid_donors},
        static=True,
        verbose=True,
    )
    print(len(aux_dataset))
    aux_loader = DataLoader(
        aux_dataset, batch_size=len(aux_dataset), collate_fn=collate_fn, shuffle=False
    )

    it = iter(aux_loader)
    d = next(it)
    patients_source = d.source["Participant ID"]
    tissues_source = d.source["Tissue"]

    print(source_tissues, target_name, len(np.unique(patients_source)))
    if len(np.unique(patients_source)) >= 25:
        # Evaluate performance when increasingly adding more tissue types
        cum_source_tissues_idxs = []
        # selected_tissues = [[tissue_dict[t]] for t in source_tissues] + [[tissue_dict[t] for t in source_tissues]]
        selected_tissues = [[t] for t in source_tissues] + [source_tissues]
        print(selected_tissues)

        for source_t, name in zip(selected_tissues, names, strict=False):
            cum_source_tissues_idxs = source_t

            # print(source_t, target_tissues)
            # Select samples from subset of individuals having all selected tissues
            aux_dataset_ = HypergraphDataset(
                adata=aux_dataset.adata_source,
                adata_target=aux_dataset.adata_target,
                obs_source={"Tissue": source_t},
                obs_target={"Tissue": target_tissues},
                static=True,
            )
            aux_loader_ = DataLoader(
                aux_dataset_, batch_size=len(aux_dataset_), collate_fn=collate_fn, shuffle=False
            )

            with torch.no_grad():
                d = next(iter(aux_loader_))
                out, node_features = forward(d, model, device, preprocess_fn=None)
                y_pred = out["px_rate"].cpu().numpy()
                y_ = d.x_target.cpu().numpy()

            gene_scores = score_fn(y_, y_pred)
            sample_scores = score_fn(y_, y_pred, sample_corr=True)
            print(
                f"Hypergraph neural network: \n Mean score per gene: {gene_scores.mean()}. Mean score per sample: {sample_scores.mean()}"
            )

            scores = sample_scores
            scores_col.extend(scores)
            source_col.extend([name] * len(scores))
            target_col.extend([target_name] * len(scores))


# In[ ]:


Counter(results_df[results_df["source"] == "Accessible All"]["target"])


# In[ ]:


results_df = pd.DataFrame({"score": scores_col, "source": source_col, "target": target_col})
mid_point = 19
tt_1 = sorted(results_df["target"].unique())[:mid_point]
tt_2 = sorted(results_df["target"].unique())[mid_point:]
results_df_1 = results_df[results_df["target"].isin(tt_1)]
results_df_2 = results_df[results_df["target"].isin(tt_2)]

sns.set(font_scale=1.6)
plt.figure(figsize=(20, 10))

plt.subplot(2, 1, 1)
sns.barplot(
    y="score", x="target", hue="source", data=results_df_1, order=np.unique(results_df_1["target"])
)
plt.legend([], [], frameon=False)
plt.xticks(rotation=45, ha="right")
plt.title("Prediction performance with accessible tissues as source")
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.subplot(2, 1, 2)
sns.barplot(
    y="score", x="target", hue="source", data=results_df_2, order=np.unique(results_df_2["target"])
)
plt.xticks(rotation=45, ha="right")

# Put the legend out of the figure
# plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

plt.title("Prediction performance with accessible tissues as source")
plt.xlabel("")
plt.ylabel("Pearson correlation")
plt.tight_layout(pad=1.0)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -1), fancybox=True, shadow=True, ncol=5)

# plt.savefig(f'figures/scores_pertissue_HYFA_accessible.pdf', bbox_inches='tight')


# In[ ]:


results_df.to_csv(f"{RESULTS_DIR}/results_pertissue_HYFA_sources_accessible.csv")


# #### Metagene-factors GSEA

# In[ ]:


library = blitz.enrichr.get_library("KEGG_2021_Human")


# In[ ]:


metagene_w = model.metagenes_encoder.encoder[0].weight.detach().cpu().numpy()
metagene_w = metagene_w.reshape((config.meta_G, -1, config.G))


# In[ ]:


n_factors = config.d_edge_attr
n_metagenes = config.meta_G

results_df = pd.DataFrame()
for factor_idx in range(n_factors):
    print(f"Factor={factor_idx}")
    for m in range(n_metagenes):
        gene_idxs = np.argsort(metagene_w[m, factor_idx, :])[::-1]
        gene_names = adata.var["Symbol"][gene_idxs].values
        gene_values = metagene_w[m, factor_idx, gene_idxs]
        signature = pd.DataFrame({0: gene_names, 1: gene_values})

        result = blitz.gsea(signature, library, permutations=2000, signature_cache=True)
        result["Factor"] = factor_idx
        result["Metagene"] = m
        results_df = pd.concat([results_df, result], axis=0)


# In[ ]:


results_df.to_csv(f"{RESULTS_DIR}/blitz_gsea_results.csv")


# In[ ]:


significant_results_df = results_df[results_df["fdr"] < 0.05]


# In[ ]:


sns.set_style("white")
plt.figure(figsize=(20, 4))
ax = plt.gca()
cmap = plt.get_cmap("tab10")

n_metagenes = 50
n_factors = 99
for factor_idx in range(n_factors):
    df = results_df[results_df["Factor"] == factor_idx]
    for x in range(n_metagenes):
        df_meta = df[df["Metagene"] == x]
        y = -np.log10(df_meta["fdr"].values)
        x_pos = [factor_idx + (x / n_metagenes)] * len(y)
        ax.scatter(x_pos, y, s=1, color=cmap(factor_idx % 10))
plt.xlabel("Factor")
plt.ylabel("$-\log_{10}(q)$")
plt.title("All human pathways (KEGG)")
plt.xticks(list(range(n_factors)))
plt.xticks(rotation=90)
plt.xlim((-1, 99))
# plt.axhline(y = -np.log10(0.05), color = 'gray', linestyle = '--', linewidth=1)
plt.savefig(f"{RESULTS_DIR}/figures/manhattan_blitzgsea.pdf", bbox_inches="tight")
# plt.savefig('overleaf/figures/manhattan_blitzgsea.png', bbox_inches='tight');


# In[ ]:


sns.set_style("white")
plt.figure(figsize=(20, 4))
ax = plt.gca()
cmap = plt.get_cmap("tab10")

n_metagenes = 50
n_factors = 99
for factor_idx in range(n_factors):
    df = results_df[results_df["Factor"] == factor_idx]
    for x in range(n_metagenes):
        df_meta = df[df["Metagene"] == x]
        y = -np.log10(df_meta["fdr"].values)
        x_pos = [x + (factor_idx / n_factors)] * len(y)
        ax.scatter(x_pos, y, s=1, color=cmap(x % 10))
plt.xlabel("Metagene")
plt.ylabel("$-\log_{10}(q)$")
plt.title("All human pathways (KEGG)")
locs = list(range(n_metagenes))
# plt.xticks(locs, rotation = 90)
plt.gca().set_xticklabels("")
plt.gca().set_xticks(np.array(locs) + 0.5, minor=True)
plt.gca().set_xticklabels([str(loc) for loc in locs], minor=True)
plt.xlim((-1, 51))
# plt.axhline(y = -np.log10(0.05), color = 'gray', linestyle = '--', linewidth=1)
# plt.savefig(f'{RESULTS_DIR}/figures/manhattan_metagenes_blitzgsea.pdf', bbox_inches='tight');
plt.savefig(f"{RESULTS_DIR}/figures/manhattan_metagenes_blitzgsea.pdf", bbox_inches="tight")
# plt.savefig('overleaf/figures/manhattan_blitzgsea.png', bbox_inches='tight');


# #### Families of pathways

# In[ ]:


def list_KEGG_human_pathways():
    lines = REST.kegg_list("pathway", "hsa").readlines()
    symbols = np.array([s.split("\t")[0].split(":")[-1] for s in lines])
    description = np.array([s.split("\t")[1].rstrip() for s in lines])
    return symbols, description


def get_pathway_class(pathway):
    pathway_file = REST.kegg_get(pathway).read()  # query and read each pathway

    pathway_class = None
    for line in pathway_file.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if section != "":
            current_section = section

        if current_section == "CLASS":
            if pathway_class is not None:
                print("Pathway belongs to more than one class")
                break
            pathway_class = line[12:]

    return pathway_class


# In[ ]:


hp, hp_desc = list_KEGG_human_pathways()


# In[ ]:


results_df_significant = results_df[results_df["fdr"] < 0.05].copy()
pathway_classes_dict = {}
for term in tqdm(np.unique(results_df_significant.index)):
    pathway_idx = np.where([term.lower() in p.lower() for p in hp_desc])[0]
    if len(pathway_idx) == 0:
        pathway_classes_dict[term] = "Unknown"
    else:
        pathway_idx = pathway_idx[0]
        pathway_code = hp[pathway_idx]
        pathway_classes_dict[term] = get_pathway_class(pathway_code)


# In[ ]:


families_dict = {k: p.split(";")[0] for k, p in pathway_classes_dict.items()}
classes_dict = {
    k: p.split(";")[1].lstrip() if len(p.split(";")) > 1 else p.split(";")[0]
    for k, p in pathway_classes_dict.items()
}


# In[ ]:


results_df_significant["class"] = results_df_significant.index.map(classes_dict)
results_df_significant["Category"] = results_df_significant.index.map(families_dict)


# In[ ]:


def unique_in_order(a):
    indexes = np.unique(a, return_index=True)[1]
    return a[np.sort(indexes)]


sorted_idxs = np.argsort(np.asarray(results_df_significant["Category"].values))

unique_in_order(np.array(results_df_significant["class"].values))


# In[ ]:


sorted_classes = []
for c in np.array(results_df_significant["class"].values)[sorted_idxs]:
    if c not in sorted_classes:
        sorted_classes.append(c)


# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 3))
sns.countplot(
    x="class",  # hue='Category',
    data=results_df_significant.reset_index(),
    # height=4,
    # aspect=4,
    ax=plt.gca(),
    order=sorted_classes,
)
# plt.gca().legend(loc='upper center', bbox_to_anchor=(0.5, -0.9), fancybox=True, shadow=True, ncol=4)
plt.xticks(rotation=90)
plt.xlabel("")
# plt.ylabel('')
plt.title("Number of enriched terms per type of pathway")
# plt.savefig(f'{RESULTS_DIR}/figures/enriched_terms_pathway_type_blitzgsea.pdf', bbox_inches='tight');


# In[ ]:


results_df_significant[results_df_significant["class"] == "Neurodegenerative disease"]


# #### FDR neurodegenerative

# In[ ]:


cutoff = 0.05
aggregated_df = pd.DataFrame()
for m in range(50):
    agg_slice_df = cast(
        pd.DataFrame,
        results_df[results_df["Metagene"] == m]
        .reset_index()[["Term", "fdr", "Factor"]]
        .set_index(["Term", "Factor"])
        .unstack(),
    )
    aggregated_df[m] = agg_slice_df.min(axis=1)


# In[ ]:


neurodegenerative_pathways = [
    "PATHWAYS OF NEURODEGENERATION",
    "AMYOTROPHIC LATERAL SCLEROSIS",
    "ALZHEIMER DISEASE",
    "PARKINSON DISEASE",
    "HUNTINGTON DISEASE",
    "PRION DISEASE",
    "SPINOCEREBELLAR ATAXIA",
]

min_fdr_per_term = aggregated_df.min(axis=1)
mean_fdr_per_term = aggregated_df.mean(axis=1)
df = cast(pd.DataFrame, aggregated_df[min_fdr_per_term < 0.05])
mask = [s in neurodegenerative_pathways for s in df.index]
df = cast(pd.DataFrame, df[mask])
fdr_mask = df.values < 0.05
df = -(df + 1e-10).apply(np.log10)


# In[ ]:


sum([s in neurodegenerative_pathways for s in results_df[results_df["fdr"] < 0.05].index])


# In[ ]:


sns.reset_orig()
x_axis = np.arange(50)
y_ticks = np.arange(df.shape[0])
x_, y_ = np.meshgrid(x_axis, y_ticks)
sizes = 20 * df.values  # 0.01/(df.values+1e-5)

sizes = sizes[fdr_mask]
x_ = x_[fdr_mask]
y_ = y_[fdr_mask]
c = df.values[fdr_mask]

plt.figure(figsize=(15, 8))
cmap = plt.get_cmap("plasma")
norm = matplotlib.colors.Normalize()
plt.scatter(x_.flatten(), y_.flatten(), s=sizes, c=c, norm=norm, cmap=cmap)
plt.yticks(ticks=y_ticks, labels=[str(label) for label in df.index.values])
cbar = plt.colorbar(fraction=0.03)
cbar.set_label("$-\log_{10}(q)$", rotation=270, labelpad=10)

minorLocator = MultipleLocator(base=1.0)
plt.gca().xaxis.set_minor_locator(minorLocator)
plt.gca().grid(which="both")
plt.gca().set_axisbelow(True)
plt.xlim(-1, 50)
plt.ylim(-0.5, 6.5)

plt.xlabel("Metagene")
plt.title("GSEA FDR for pathways of neurodegeneration (KEGG)")
plt.savefig(
    f"{RESULTS_DIR}/figures/fdr_scatter_metagenes_neurodegeneration_blitzgsea.png",
    bbox_inches="tight",
)
# #### FDR signaling

# In[ ]:


cutoff = 0.05
aggregated_df = pd.DataFrame()
for factor_idx in range(99):
    agg_slice_df = cast(
        pd.DataFrame,
        results_df[results_df["Factor"] == factor_idx]
        .reset_index()[["Term", "fdr", "Metagene"]]
        .set_index(["Term", "Metagene"])
        .unstack(),
    )
    aggregated_df[factor_idx] = agg_slice_df.min(axis=1)


# In[ ]:


signaling_pathways = [
    c
    for c, f in classes_dict.items()
    if f == "Signaling molecules and interaction"
    or f == "Signal transduction"
    or f == "Membrane transport"
]

min_fdr_per_term = aggregated_df.min(axis=1)
mean_fdr_per_term = aggregated_df.mean(axis=1)
df = cast(pd.DataFrame, aggregated_df[min_fdr_per_term < 0.05])
mask = [s in signaling_pathways for s in df.index]
df = cast(pd.DataFrame, df[mask])
fdr_mask = df.values < 0.05
df = -(df + 1e-10).apply(np.log10)


# In[ ]:


sum([s in signaling_pathways for s in results_df[results_df["fdr"] < 0.05].index])


# In[ ]:


sns.reset_orig()
x_axis = np.arange(99)
y_ticks = np.arange(df.shape[0])
x_, y_ = np.meshgrid(x_axis, y_ticks)
sizes = 50 * df.values  # 0.01/(df.values+1e-5)

sizes = sizes[fdr_mask]
x_ = x_[fdr_mask]
y_ = y_[fdr_mask]
c = df.values[fdr_mask]

plt.figure(figsize=(25, 15))
cmap = plt.get_cmap("plasma")
norm = matplotlib.colors.Normalize()
plt.scatter(x_.flatten(), y_.flatten(), s=sizes, c=c, norm=norm, cmap=cmap)
plt.yticks(ticks=y_ticks, labels=[str(label) for label in df.index.values])
cbar = plt.colorbar(fraction=0.03)
cbar.set_label("$-\log_{10}(q)$", rotation=270, labelpad=10)

minorLocator = MultipleLocator(base=1.0)
plt.gca().xaxis.set_minor_locator(minorLocator)
plt.gca().grid(which="both")
plt.gca().set_axisbelow(True)
plt.xlim(-1, 99)
# plt.ylim(-0.5, 6.5)

plt.xlabel("Metagene")
plt.title("GSEA FDR for signaling pathways (KEGG)")
plt.savefig(
    f"{RESULTS_DIR}/figures/fdr_scatter_factors_signaling_blitzgsea.png", bbox_inches="tight"
)
# #### Neurodegenerative

# In[ ]:


subject_df = pd.read_csv(METADATA_FILE, delimiter="\t")
subject_df = subject_df.set_index("SUBJID")


# In[ ]:


aux_val_dataset_ = HypergraphDataset(adata, obs_source={"Tissue": ["Brain_Cortex"]})
aux_val_loader_ = DataLoader(
    aux_val_dataset_, batch_size=len(aux_val_dataset_), collate_fn=collate_fn, shuffle=False
)

# Compute predictions and score
model.eval()
with torch.no_grad():
    d = next(iter(aux_val_loader_))
    d = d.to(device)
    x_source = d.x_source
    x_source = model.encode_metagenes(x_source)

x_source = x_source.detach().cpu().numpy()
participant_idxs = d.source["Participant ID"].detach().cpu().numpy()
participant_idxs = [aux_val_dataset_.donor_map[p] for p in participant_idxs]


# In[ ]:


with open("results/encoded_metagenes_brain_cortex.npy", "wb") as fh:
    np.save(fh, x_source)

with open("results/participant_idxs_brain_cortex.npy", "wb") as fh:
    np.save(fh, participant_idxs)


# In[ ]:


factor_id = 95
metagene_idx = 11
gene_idxs = np.argsort(metagene_w[metagene_idx, factor_id, :])[::-1]
gene_names = adata.var["Symbol"][gene_idxs].values
gene_values = metagene_w[metagene_idx, factor_id, gene_idxs]
signature = pd.DataFrame({0: gene_names, 1: gene_values})
# result = blitz.gsea(signature, library, permutations=100)

df = results_df[(results_df["Factor"] == factor_id) & (results_df["Metagene"] == metagene_idx)]
fig = blitz.plot.top_table(signature, library, df, n=15)


# In[ ]:


fig = blitz.plot.running_sum(signature, df.index[0], library, result=df, compact=False)
plt.suptitle(f"Metagene {metagene_idx}, factor {factor_id}")
fig.set_size_inches((5, 7))


# In[ ]:


key = "MHALZDMT"
x = x_source[:, 11, :]

x_2d = umap.UMAP(random_state=0).fit_transform(x)

plt.figure(figsize=(4, 4))
y = subject_df.loc[participant_idxs][key]
plt.scatter(x_2d[y == 0, 0], x_2d[y == 0, 1], s=20, cmap=plt.get_cmap("summer"), label="Control")
plt.gca().scatter(
    x_2d[y == 1, 0], x_2d[y == 1, 1], s=50, marker="^", cmap=plt.get_cmap("summer"), label="ALZDMT"
)
plt.legend(loc="upper left")
plt.title("Alzheimer or dementia (brain cortex)")
plt.xlabel("Metagene 11, UMAP 1")
plt.ylabel("Metagene 11, UMAP 2")
plt.savefig(f"{RESULTS_DIR}/figures/metagenes_ALZDMT_cortex.pdf", bbox_inches="tight")
Counter(y), Counter(subject_df[key])


# In[ ]:


significant_results_df.reset_index().query("Term == 'AMYOTROPHIC LATERAL SCLEROSIS'").sort_values(
    by="fdr"
)


# In[ ]:


# Hypergraph baseline (all tissues)
# Select same set of individuals
aux_val_dataset_ = HypergraphDataset(adata, obs_source={"Tissue": ["Brain_Spinal_cord"]})
aux_val_loader_ = DataLoader(
    aux_val_dataset_, batch_size=len(aux_val_dataset_), collate_fn=collate_fn, shuffle=False
)

# Compute predictions and score
model.eval()
with torch.no_grad():
    d = next(iter(aux_val_loader_))
    d = d.to(device)
    x_source = d.x_source
    x_source = model.encode_metagenes(x_source)

x_source = x_source.detach().cpu().numpy()
participant_idxs = d.source["Participant ID"].detach().cpu().numpy()
participant_idxs = [aux_val_dataset_.donor_map[p] for p in participant_idxs]


# In[ ]:


with open("results/encoded_metagenes_brain_spinal_cord.npy", "wb") as fh:
    np.save(fh, x_source)

with open("results/participant_idxs_brain_spinal_cord.npy", "wb") as fh:
    np.save(fh, participant_idxs)


# In[ ]:


sns.set_style("whitegrid")
key = "MHALS"  # 'MHALS'
x = x_source[:, 11, :]

x_2d = umap.UMAP(random_state=0).fit_transform(x)

plt.figure(figsize=(4, 4))
y = subject_df.loc[participant_idxs][key]
plt.scatter(x_2d[y == 0, 0], x_2d[y == 0, 1], s=20, cmap=plt.get_cmap("summer"), label="Control")
plt.gca().scatter(
    x_2d[y == 1, 0], x_2d[y == 1, 1], s=50, marker="^", cmap=plt.get_cmap("summer"), label=key
)
plt.legend(loc="upper left")
plt.title("Amyotrophic Lateral Sclerosis (spinal cord)")
plt.xlabel("Metagene 11, UMAP 1")
plt.ylabel("Metagene 11, UMAP 2")
plt.savefig(f"{RESULTS_DIR}/figures/metagenes_MHALS_spinalcord.pdf", bbox_inches="tight")
Counter(y), Counter(subject_df[key])


# In[ ]:


factor_id = 95
metagene_idx = 11
gene_idxs = np.argsort(metagene_w[metagene_idx, factor_id, :])[::-1]
gene_names = adata.var["Symbol"][gene_idxs].values
gene_values = metagene_w[metagene_idx, factor_id, gene_idxs]
signature = pd.DataFrame({0: gene_names, 1: gene_values})
# result = blitz.gsea(signature, library, permutations=100)

df = results_df[(results_df["Factor"] == factor_id) & (results_df["Metagene"] == metagene_idx)]
fig = blitz.plot.top_table(signature, library, df, n=15)
plt.title(f"Metagene {metagene_idx}, factor {factor_id}")
fig.set_size_inches((5, 7))
plt.savefig(f"{RESULTS_DIR}/figures/metagenes_ALS_ALZDMT_top_plot.pdf", bbox_inches="tight")


# In[ ]:


fig = blitz.plot.running_sum(signature, df.index[0], library, result=df, compact=False)
plt.suptitle(f"Metagene {metagene_idx}, factor {factor_id}")
fig.set_size_inches((5, 7))
plt.savefig(f"{RESULTS_DIR}/figures/metagenes_ALS_running_sum.pdf", bbox_inches="tight")


# In[ ]:


fig = blitz.plot.running_sum(signature, df.index[1], library, result=df, compact=False)
plt.suptitle(f"Metagene {metagene_idx}, factor {factor_id}")
fig.set_size_inches((5, 7))
plt.savefig(f"{RESULTS_DIR}/figures/metagenes_Alzheimer_running_sum.pdf", bbox_inches="tight")


# #### Store predictions

# In[ ]:


def create_dataframe(participant_ids, tissue_ids, expression, donor_map, tissue_dict_inv, symbols):
    participant_ids = np.concatenate(participant_ids, axis=0)
    tissue_ids = np.concatenate(tissue_ids, axis=0)
    expression = np.concatenate(expression, axis=0)
    df_metadata = pd.DataFrame(
        {
            "Participant ID": [donor_map[p] for p in participant_ids],
            "Tissue": [tissue_dict_inv[t] for t in tissue_ids],
        }
    )
    df = pd.DataFrame(expression, columns=symbols)
    df = pd.concat([df_metadata, df], axis=1)
    df = df.set_index("Participant ID")
    return df


# In[ ]:


dataset: Any = HypergraphDataset(adata, static=True)


# In[ ]:


model.eval()
# df_imputed = pd.DataFrame({'Participant ID': [], 'Tissue ID': [], })
source_participant_ids: list[np.ndarray] = []
source_tissue_ids: list[np.ndarray] = []
source_expression: list[np.ndarray] = []
target_participant_ids: list[np.ndarray] = []
target_tissue_ids: list[np.ndarray] = []
target_expression: list[np.ndarray] = []
for i in tqdm(range(len(dataset))):
    d = dataset[i]
    # Set target tissues to missing tissues
    d.target["Tissue"] = torch.tensor(
        [t for t in np.arange(len(tissue_dict)) if t not in d.source["Tissue"]]
    )
    d.target["Participant ID"] = (
        torch.zeros_like(d.target["Tissue"]) + d.source["Participant ID"][0]
    )
    d.x_target = torch.tensor([-1])  # Unused

    # Make predictions
    with torch.no_grad():
        out, node_features = forward(d, model, device, preprocess_fn=None)
        y_pred = out["px_rate"]

    # Store
    source_participant_ids.append(d.source["Participant ID"].cpu().numpy() + i)
    source_tissue_ids.append(d.source["Tissue"].cpu().numpy())
    source_expression.append(d.x_source.cpu().numpy())
    target_participant_ids.append(d.target["Participant ID"].cpu().numpy() + i)
    target_tissue_ids.append(d.target["Tissue"].cpu().numpy())
    target_expression.append(y_pred.cpu().numpy())

# Store data in dataframes
df_imputed = create_dataframe(
    target_participant_ids,
    target_tissue_ids,
    target_expression,
    donor_map=dataset.donor_map,
    tissue_dict_inv=tissue_dict_inv,
    symbols=adata.var["Symbol"],
)
df_observed = create_dataframe(
    source_participant_ids,
    source_tissue_ids,
    source_expression,
    donor_map=dataset.donor_map,
    tissue_dict_inv=tissue_dict_inv,
    symbols=adata.var["Symbol"],
)


# In[ ]:


df_observed.to_csv(f"{RESULTS_DIR}/observed_normalised.csv")
df_imputed.to_csv(f"{RESULTS_DIR}/imputed_normalised.csv")


# #### Tissue to tissue network

# In[ ]:


t2t_scores = np.load(f"{RESULTS_DIR}/t2t_scores.npy")


# In[ ]:


score_fn = pearson_correlation_score
t2t_scores_dict: dict[str, dict[str, dict[str, float]]] = {}
unseen_mask = np.logical_or(val_mask, test_mask)

for st in tissue_dict:
    st2t_scores_dict = {}
    for tt in tqdm(tissue_dict):
        print(st, "->", tt)
        if st in t2t_scores_dict and tt in t2t_scores_dict[st]:
            continue

        # Name source and target tissues
        source_name = st.replace("_", " ")
        target_name = tt.replace("_", " ")
        # print(tt)

        # Create datasets
        aux_dataset = HypergraphDataset(
            adata[unseen_mask], obs_source={"Tissue": [st]}, obs_target={"Tissue": [tt]}
        )
        source_donor_ids = aux_dataset.adata_source.obs["Participant ID"]
        target_donor_ids = aux_dataset.adata_target.obs["Participant ID"]
        assert (source_donor_ids.values == target_donor_ids.values).all()

        if len(aux_dataset) < 10:
            print("Less than 10 samples", st, tt)
            continue

        # Hypergraph baseline
        aux_loader = DataLoader(
            aux_dataset,
            batch_size=len(aux_dataset),
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
        )

        # Compute predictions and score
        model.eval()
        with torch.no_grad():
            d = next(iter(aux_loader))

            out, node_features = forward(d, model, device, preprocess_fn=None)
            y_test_pred = (
                out["px_rate"].cpu().numpy()
            )  # torch.distributions.normal.Normal(loc=out['px_rate'], scale=out['px_r']).mean.cpu().numpy()
            y_test_ = d.x_target.cpu().numpy()

        sample_scores = score_fn(y_test_, y_test_pred, sample_corr=True)
        gene_scores = score_fn(y_test_, y_test_pred, sample_corr=False)

        # Append results
        st2t_scores_dict[tt] = {
            "gene_scores": gene_scores.mean(),
            "sample_scores": sample_scores.mean(),
        }
        del aux_dataset
        del aux_loader

    t2t_scores_dict[st] = st2t_scores_dict


# In[ ]:


t2t_scores = np.zeros((len(tissue_dict), len(tissue_dict)))
for i, st in enumerate(tissue_dict.keys()):
    for j, tt in enumerate(tissue_dict.keys()):
        if tt in t2t_scores_dict[st]:
            t2t_scores[i, j] = t2t_scores_dict[st][tt]["gene_scores"]


# In[ ]:


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

fig = plt.figure(figsize=(8, 8))
threshold = 0.4
t2t_scores_ = t2t_scores * (1 - np.eye(t2t_scores.shape[0]))
G = nx.from_numpy_array(t2t_scores_ > threshold, create_using=nx.DiGraph)
G = nx.relabel_nodes(G, tissue_dict_inv)
G.remove_nodes_from(list(nx.isolates(G)))
pos = nx.circular_layout(G)  #
# pos = nx.spring_layout(G, pos=pos, k=0.1, iterations=2)

edge_weights = np.array([t2t_scores[tissue_dict[u], tissue_dict[v]] for u, v in G.edges])
# edge_weights = 0.2 + 2*(edge_weights - np.min(edge_weights))/ (np.max(edge_weights) - np.min(edge_weights))
node_size = np.array([G.degree[u] * 10 for u in G.nodes])
labels = {
    k: k.replace("_", " ").replace("Brain ", "").replace(" Omentum", "").replace(" Tissue", "")
    for k in G.nodes
}

color_map = [colors[tissue_dict[t]] for t in G.nodes]

# nx.draw(G, pos=pos, with_labels = True, width=weights)  # node_size=[d[k]*100 for k in d]
nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, edge_color="gray")
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=color_map, alpha=0.9)

labels_pos = {}
n = len(pos)
shiftval = 0.12
for i, (k, _v) in enumerate(pos.items()):
    labels_pos[k] = pos[k] + np.sqrt(len(labels[k])) * np.array(
        [shiftval * np.cos(i * 2 * np.pi / n), shiftval * np.sin(i * 2 * np.pi / n)]
    )

    if k == "Esophagus_Muscularis":
        labels_pos[k] += (-0.07, -0.01)
    elif k == "Esophagus_Mucosa":
        labels_pos[k] += (-0.06, -0.0)
    elif k == "Esophagus_Gastro":
        labels_pos[k] += (-0.06, +0.01)
    elif k == "Colon_Transverse":
        labels_pos[k] += (-0.05, +0.02)
    elif k == "Colon_Sigmoid":
        labels_pos[k] += (-0.04, +0.03)
    elif k == "Breast_Mammary_Tissue":
        labels_pos[k] += (-0.03, +0.04)


text = nx.draw_networkx_labels(G, labels_pos, labels, font_size=12)
for i, (_, t) in enumerate(text.items()):
    angle = 360 * i / len(text.items())
    if np.cos(angle * (np.pi / 180)) < 0:
        angle = angle + 180
    t.set_rotation(angle)

plt.gca().axis("off")

marginval = 0.9
x1, x2, y1, y2 = plt.axis()
plt.axis((x1 - marginval, x2 + marginval, y1 - marginval, y2 + marginval))
plt.tight_layout()
plt.savefig(f"figures/tissue_to_tissue_network_{threshold}cutoff_pergene.pdf", bbox_inches="tight")
