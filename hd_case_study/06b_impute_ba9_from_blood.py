#!/usr/bin/env python3
"""
06b_impute_ba9_from_blood.py -- impute BA9 cortex from SRP032279 whole blood via pretrained HYFA.
Bridges DeepSAGE blood into HYFA's GTEx input space by the same per-gene inverse-normal transform
(rank-based, platform-robust), reindexes onto HYFA's panel (missing genes -> 0), and feeds each
sample as a Whole_Blood source with a dummy Brain_Frontal_Cortex target. Locates the HYFA repo and
runs from it. SRP032279 has no sex -> sex covariate set to 0.

Inputs : prep/hd_blood_counts_by_symbol.csv, prep/hd_blood_pheno.csv,
         <hyfa>/data/{GTEX_data.csv, normalised_model_default.pth, annotations txt}
Output : prep/hd_blood_imputed_ba9.csv  (blood sample_id x gene, HYFA inverse-normal space)
Usage  : python 06b_impute_ba9_from_blood.py [--source Whole_Blood] [--target Brain_Frontal_Cortex] [--hyfa-root PATH]
"""
import argparse, os, sys
import numpy as np
import pandas as pd


def find_hyfa_root(explicit=None):
    def is_root(d):
        return (os.path.isfile(os.path.join(d, "train_gtex.py"))
                and os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "configs")))
    if explicit:
        d = os.path.abspath(explicit); return d if is_root(d) else None
    starts = []
    _f = globals().get("__file__")
    if _f: starts.append(os.path.dirname(os.path.abspath(_f)))
    starts.append(os.getcwd())
    for s in starts:
        d = s
        for _ in range(4):
            if is_root(d): return d
            nd = os.path.dirname(d)
            if nd == d: break
            d = nd
    for s in starts:
        for root, dirs, files in os.walk(s):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".ipynb_checkpoints")]
            if "train_gtex.py" in files and os.path.isdir(os.path.join(root, "src")) \
               and os.path.isdir(os.path.join(root, "configs")):
                return root
    return None


def inverse_normal(M):
    """Per-column (per-gene) inverse-normal transform (Blom). M: samples x genes."""
    from scipy.stats import rankdata, norm
    n = M.shape[0]; out = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        r = rankdata(M[:, j], method="average")
        out[:, j] = norm.ppf((r - 0.375) / (n + 0.25))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Whole_Blood")
    ap.add_argument("--target", default="Brain_Frontal_Cortex")
    ap.add_argument("--hyfa-root", default=None)
    a = ap.parse_args()

    orig = os.getcwd()
    root = find_hyfa_root(a.hyfa_root)
    if root is None:
        sys.exit("HYFA repo not found; pass --hyfa-root")
    prep = os.path.join(orig, "prep")
    counts_f = os.path.join(prep, "hd_blood_counts_by_symbol.csv")
    pheno_f = os.path.join(prep, "hd_blood_pheno.csv")
    for f in (counts_f, pheno_f):
        if not os.path.exists(f):
            sys.exit("missing input (run 02 first): " + f)
    out_f = os.path.join(prep, "hd_blood_imputed_ba9.csv")
    data_csv = os.path.join(root, "data", "GTEX_data.csv")
    model_p = os.path.join(root, "data", "normalised_model_default.pth")
    for f in (data_csv, model_p):
        if not os.path.exists(f):
            sys.exit("missing HYFA data: " + f)

    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    # ---- HYFA load (verified shims from 02a) ----
    import torch, wandb
    import scipy.sparse as sp
    import anndata as ad
    def _load(pth):
        try:    return torch.load(pth, map_location="cpu", weights_only=False)
        except TypeError: return torch.load(pth, map_location="cpu")
    import torch_scatter as _ts
    def _long(fn):
        def w(src, index, *ar, **kw):
            if index.dtype != torch.long: index = index.long()
            return fn(src, index, *ar, **kw)
        return w
    for nm in ("scatter_max", "scatter_add", "scatter_mean", "scatter_min", "scatter_std", "scatter_softmax"):
        if hasattr(_ts, nm): setattr(_ts, nm, _long(getattr(_ts, nm)))
    from train_gtex import (GTEx_v8_normalised_adata, HypergraphDataset,
                            HypergraphNeuralNet, Data, DataLoader)
    from src.train_utils import forward

    wandb.init(project="multitissue_imputation", config="configs/default.yaml", mode="disabled")
    config = wandb.config
    print("loading GTEx adata (panel + tissue map + config) ...")
    gtex = GTEx_v8_normalised_adata(file=data_csv)
    device = torch.device("cpu")
    config.update({"static_node_types": {
        "Tissue": (len(gtex.obs["Tissue_idx"].unique()), config.d_tissue),
        "metagenes": (config.meta_G, config.d_gene)}}, allow_val_change=True)
    config.update({"dynamic_node_types": {
        "Participant ID": (len(gtex.obs["Participant ID"].unique()), config.d_patient)}},
        allow_val_change=True)
    config.G = gtex.shape[-1]
    model = HypergraphNeuralNet(config).to(device)
    model.load_state_dict(_load(model_p)); model.eval()
    print("pretrained HYFA loaded.")

    panel = list(np.asarray(gtex.var["Symbol"].values))
    tdict = dict(gtex.uns["Tissue_dict"])
    for t in (a.source, a.target):
        if t not in tdict:
            sys.exit("tissue '%s' not in GTEx tissue_dict; available brain/blood: %s"
                     % (t, [k for k in tdict if "Brain" in k or "Blood" in k]))
    src_idx, tgt_idx = int(tdict[a.source]), int(tdict[a.target])
    print("source=%s(idx %d)  target=%s(idx %d)  panel=%d genes"
          % (a.source, src_idx, a.target, tgt_idx, len(panel)))

    # ---- blood -> inverse-normal -> panel ----
    cnt = pd.read_csv(counts_f); cnt = cnt.set_index(cnt.columns[0])   # genes x samples
    samples = list(cnt.columns)
    lib = cnt.sum(0).replace(0, np.nan)
    cpm = (cnt.divide(lib, axis=1) * 1e6).T.values                    # samples x genes (blood gene order)
    int_blood = inverse_normal(cpm)                                   # samples x genes
    blood_genes = list(cnt.index)
    gpos = {g: i for i, g in enumerate(blood_genes)}
    panel_mat = np.zeros((len(samples), len(panel)), dtype=float)     # panel genes absent in blood -> 0
    hit = 0
    for j, g in enumerate(panel):
        if g in gpos:
            panel_mat[:, j] = int_blood[:, gpos[g]]; hit += 1
    print("panel genes present in blood: %d / %d" % (hit, len(panel)))

    ph = pd.read_csv(pheno_f).set_index("sample_id").reindex(samples)
    age = pd.to_numeric(ph.get("attr_age"), errors="coerce").fillna(
        pd.to_numeric(ph.get("attr_age"), errors="coerce").median()).values
    feat = np.stack([age / 100.0, np.zeros(len(samples))], axis=-1)   # (age/100, sex=0)

    # ---- external adata: blood source rows + dummy BA9 target rows ----
    N = len(samples)
    X = np.vstack([panel_mat, np.zeros((N, len(panel)))])             # 2N x G (blood; dummy cortex)
    obs = pd.DataFrame({
        "Participant ID": list(samples) + list(samples),
        "Participant ID_dyn": list(samples) + list(samples),
        "Tissue": [a.source] * N + [a.target] * N,
        "Tissue_idx": [src_idx] * N + [tgt_idx] * N,
    })
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs)
    adata.var_names = panel
    adata.layers["x"] = sp.csr_matrix(X)
    adata.obsm["Participant ID_feat"] = np.vstack([feat, feat])

    ds = HypergraphDataset(adata,
                           obs_source={"Tissue": [a.source], "Participant ID": samples},
                           obs_target={"Tissue": [a.target], "Participant ID": samples},
                           static=True)
    loader = DataLoader(ds, batch_size=len(ds), collate_fn=Data.from_datalist, shuffle=False)
    d = next(iter(loader))
    with torch.no_grad():
        out, _ = forward(d, model, device, preprocess_fn=None)
        imputed = out["px_rate"].cpu().numpy()                        # nb_donors x G, aligned to sorted donors

    order = sorted(samples)                                            # HypergraphDataset sorts donor_ids
    df = pd.DataFrame(imputed, index=order, columns=panel)
    df.index.name = "sample_id"
    df.to_csv(out_f)
    print("\nimputed BA9: %d samples x %d genes -> %s" % (df.shape[0], df.shape[1], out_f))
    print("(HYFA inverse-normal space; 06b_classify refits conditionals in this same space.)")


if __name__ == "__main__":
    main()
