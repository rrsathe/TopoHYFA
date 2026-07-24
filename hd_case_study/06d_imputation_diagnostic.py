#!/usr/bin/env python3
"""
06d_imputation_diagnostic.py -- why does the imputed-cortex expression baseline sit at chance?
Characterizes HYFA's blood->cortex fidelity on IN-DISTRIBUTION GTEx whole blood (no platform
gap), over the module genes:
  Part A  per-gene recoverability (imputed vs true GTEx cortex) and variance ratio
          (var imputed / var true) -- does HYFA preserve per-donor spread or mean-revert?
  Part B  per-gene variance of the DeepSAGE-imputed HD cortex vs the GTEx-imputed cortex
          (both are HYFA px_rate) -- does off-distribution DeepSAGE input collapse HYFA further?
Separates "DeepSAGE platform gap" (recovers in-distribution, collapses on DeepSAGE) from
"healthy-trained map is normal-locked" (mean-reverts even in-distribution).

Runs from anywhere (locates the HYFA repo, like 02a/06b_impute).
Inputs : <hyfa>/data/{GTEX_data.csv, normalised_model_default.pth, annotations txt},
         grn/grn_lcc_edges.csv, prep/hd_blood_imputed_ba9.csv
Output : grn/imputation_diagnostic.txt
Usage  : python 06d_imputation_diagnostic.py [--source Whole_Blood] [--target Brain_Frontal_Cortex] [--min-paired 25]
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


def col_pearson(A, B, eps=1e-9):
    A = A - A.mean(0, keepdims=True); B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0); den = np.sqrt((A ** 2).sum(0)) * np.sqrt((B ** 2).sum(0))
    r = np.full(A.shape[1], np.nan); ok = den > eps; r[ok] = num[ok] / den[ok]
    return r


def med(x):
    x = x[np.isfinite(x)]; return float(np.median(x)) if x.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Whole_Blood")
    ap.add_argument("--target", default="Brain_Frontal_Cortex")
    ap.add_argument("--min-paired", type=int, default=25)
    ap.add_argument("--hyfa-root", default=None)
    a = ap.parse_args()

    orig = os.getcwd()
    root = find_hyfa_root(a.hyfa_root)
    if root is None:
        sys.exit("HYFA repo not found; pass --hyfa-root")
    edges_f = os.path.join(orig, "grn", "grn_lcc_edges.csv")
    deep_f = os.path.join(orig, "prep", "hd_blood_imputed_ba9.csv")
    out_f = os.path.join(orig, "grn", "imputation_diagnostic.txt")
    data_csv = os.path.join(root, "data", "GTEX_data.csv")
    model_p = os.path.join(root, "data", "normalised_model_default.pth")
    for f in (edges_f, deep_f, data_csv, model_p):
        if not os.path.exists(f):
            sys.exit("missing input: " + f)

    os.chdir(root)
    if root not in sys.path: sys.path.insert(0, root)

    import torch, wandb
    def _load(p):
        try:    return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError: return torch.load(p, map_location="cpu")
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
    print("loading GTEx adata ...")
    adata = GTEx_v8_normalised_adata(file=data_csv)
    device = torch.device("cpu")
    config.update({"static_node_types": {
        "Tissue": (len(adata.obs["Tissue_idx"].unique()), config.d_tissue),
        "metagenes": (config.meta_G, config.d_gene)}}, allow_val_change=True)
    config.update({"dynamic_node_types": {
        "Participant ID": (len(adata.obs["Participant ID"].unique()), config.d_patient)}},
        allow_val_change=True)
    config.G = adata.shape[-1]
    model = HypergraphNeuralNet(config).to(device)
    model.load_state_dict(_load(model_p)); model.eval()
    print("HYFA loaded.")

    tissues = list(adata.obs["Tissue"].unique())
    for t in (a.source, a.target):
        if t not in tissues:
            sys.exit("tissue '%s' not present; brain/blood available: %s"
                     % (t, [x for x in tissues if "Brain" in x or "Blood" in x]))
    panel = list(np.asarray(adata.var["Symbol"].values)); pidx = {g: i for i, g in enumerate(panel)}

    # GTEx donors with both blood and frontal cortex -> impute cortex from blood
    donor_tissues = adata.obs.groupby("Participant ID")["Tissue"].agg(set).to_dict()
    paired = [d for d, ts in donor_tissues.items() if a.source in ts and a.target in ts]
    if len(paired) < a.min_paired:
        sys.exit("only %d blood+cortex GTEx donors (< %d)" % (len(paired), a.min_paired))
    ds = HypergraphDataset(adata,
                           obs_source={"Tissue": [a.source], "Participant ID": paired},
                           obs_target={"Tissue": [a.target], "Participant ID": paired}, static=True)
    d = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=Data.from_datalist, shuffle=False)))
    with torch.no_grad():
        out, _ = forward(d, model, device, preprocess_fn=None)
        pred = out["px_rate"].cpu().numpy(); true = d.x_target.cpu().numpy()   # GTEx-imputed, true GTEx cortex
    print("GTEx blood->cortex imputed for %d paired donors" % len(paired))

    # module = induced subgraph genes present in the HYFA panel
    ed = pd.read_csv(edges_f)
    mod = sorted({g for pair in zip(ed["source"], ed["target"]) for g in pair
                  if isinstance(g, str)} & set(panel))
    mcols = np.array([pidx[g] for g in mod])
    print("module genes in panel: %d" % len(mod))

    # Part A: recoverability + variance ratio (in-distribution GTEx)
    r_all = col_pearson(true, pred)
    vr_all = pred.var(0) / np.where(true.var(0) > 1e-12, true.var(0), np.nan)
    r_mod, vr_mod = r_all[mcols], vr_all[mcols]

    # Part B: DeepSAGE-imputed vs GTEx-imputed per-gene variance (both HYFA px_rate)
    deep = pd.read_csv(deep_f).set_index("sample_id")
    deep = deep.reindex(columns=panel)                 # align to panel order
    deep_var = deep.values.var(0)
    gtex_var = pred.var(0)
    ratio_all = deep_var / np.where(gtex_var > 1e-12, gtex_var, np.nan)
    ratio_mod = ratio_all[mcols]

    lines = [
        "IMPUTATION DIAGNOSTIC -- why the imputed-cortex baseline is at chance",
        "source=%s  target=%s  GTEx paired donors=%d  module genes in panel=%d"
        % (a.source, a.target, len(paired), len(mod)),
        "(GTEx is in-distribution for HYFA; paired donors include training donors, but the",
        " individual node is inferred from the source, so this is a fidelity ceiling.)",
        "",
        "PART A  HYFA on in-distribution GTEx blood -> cortex:",
        "  per-gene recoverability r (imputed vs true):  module median=%.3f | all-gene median=%.3f"
        % (med(r_mod), med(r_all)),
        "  variance ratio var(imputed)/var(true):        module median=%.3f | all-gene median=%.3f"
        % (med(vr_mod), med(vr_all)),
        "    (ratio << 1 = HYFA mean-reverts: imputed cortex has far less per-donor spread than real)",
        "",
        "PART B  DeepSAGE-imputed HD cortex vs GTEx-imputed cortex (both HYFA outputs):",
        "  per-gene variance ratio var(DeepSAGE-imputed)/var(GTEx-imputed):  module median=%.3f | all=%.3f"
        % (med(ratio_mod), med(ratio_all)),
        "    (ratio << 1 = off-distribution DeepSAGE input collapses HYFA's output further)",
        "",
        "READ:",
    ]
    rm, vm, dm = med(r_mod), med(vr_mod), med(ratio_mod)
    if rm > 0.3 and dm < 0.7:
        lines += [
            "  HYFA recovers the module genes from IN-DISTRIBUTION blood (r=%.2f) but its output" % rm,
            "  collapses on the off-distribution DeepSAGE input (%.2fx the GTEx-imputed variance)." % dm,
            "  -> the chance-level baseline is the DeepSAGE PLATFORM GAP, not the map itself.",
            "     A matched-platform (TruSeq/GTEx-like) HD whole-blood cohort would likely restore signal.",
        ]
    elif vm < 0.5:
        lines += [
            "  HYFA mean-reverts even on in-distribution blood (variance ratio=%.2f, r=%.2f): it images" % (vm, rm),
            "  an average cortex and washes out per-donor variation regardless of input.",
            "  -> the map is NORMAL-LOCKED; the chance baseline is inherent, not fixable by better blood.",
        ]
    else:
        lines += [
            "  module r=%.2f, variance ratio=%.2f, DeepSAGE/GTEx variance=%.2f -- mixed; inspect the" % (rm, vm, dm),
            "  per-gene numbers before concluding platform-gap vs normal-locked.",
        ]
    pergene_f = os.path.join(orig, "grn", "imputation_diagnostic_pergene.csv")
    pd.DataFrame({"gene": mod,
                  "recoverability_r": np.round(r_all[mcols], 4),
                  "var_ratio_imp_over_true": np.round(vr_all[mcols], 4),
                  "deepsage_over_gtex_var": np.round(ratio_all[mcols], 4)}).to_csv(pergene_f, index=False)
    with open(out_f, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines)); print("\n-> " + out_f)


if __name__ == "__main__":
    main()
