#!/usr/bin/env python
"""
02a_gtex_cortex_recoverability.py   [run order: 02a]

GTEx-only KILL-SWITCH, run BEFORE any HD data or module work.

Question: does the pretrained HYFA recover CORTEX expression FROM BLOOD at all,
per gene, on in-distribution adult GTEx? HD's tissue leg is BA9 frontal cortex; if
cortex is globally decoupled from blood here, the HD imputation leg -- and any GRN
lift on top of it -- has nothing to work with. Fail here and you stop before
spending the whole HD build.

Reuses the verified HYFA invocation from the prior run (torch_scatter long-index
guard, weights_only-compat load, forward()/px_rate path, 'Whole_Blood' naming) but
removes the module dependency (no module exists at 02a) and measures per-gene
recoverability over ALL genes. Every blood-paired tissue is scored so the cortex
number is interpretable -- cortex-specific decoupling vs a global setup failure.

Run from ANYWHERE (project root is fine): the script locates the HYFA repo (the
folder with train_gtex.py + src/ + configs/) and switches into it, since HYFA's
imports and its hardcoded data/ + configs/ paths are all CWD-relative. Override
auto-detection with --hyfa-root if needed.

    python 02a_gtex_cortex_recoverability.py

Requires in <hyfa_root>/data/ (README downloads):
    GTEX_data.csv
    GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt
    normalised_model_default.pth

Outputs (under --out, default ./recoverability/ relative to where you launch):
    recoverability_by_tissue.csv   per-tissue: n_paired, median/mean per-gene r, frac>thr
    pergene_<TISSUE>.csv           per-gene r for each brain tissue (intersect the HD module here later)
    gate_verdict.json              cortex numbers + recoverable-gate flag
    recoverability_console.txt

NOTE: paired donors include GTEx TRAINING donors, so these r are an OPTIMISTIC
ceiling -- the correct bias for a kill-switch (if cortex fails even here, it fails).
For an honest held-out number later, restrict to data/splits/gtex_test.txt.

RECORD: if it errors, fix and re-run.
"""
import argparse, csv, json, os, sys
import numpy as np


def find_hyfa_root(explicit=None):
    """Locate the HYFA repo: a dir containing train_gtex.py + src/ + configs/."""
    def is_root(d):
        return (os.path.isfile(os.path.join(d, "train_gtex.py"))
                and os.path.isdir(os.path.join(d, "src"))
                and os.path.isdir(os.path.join(d, "configs")))
    if explicit:
        d = os.path.abspath(explicit)
        return d if is_root(d) else None
    starts = []
    _f = globals().get("__file__")
    if _f: starts.append(os.path.dirname(os.path.abspath(_f)))
    starts.append(os.getcwd())
    for s in starts:                       # check start dirs + a few ancestors
        d = s
        for _ in range(4):
            if is_root(d): return d
            nd = os.path.dirname(d)
            if nd == d: break
            d = nd
    for s in starts:                       # otherwise walk downward
        for root, dirs, files in os.walk(s):
            dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".ipynb_checkpoints")]
            if "train_gtex.py" in files and os.path.isdir(os.path.join(root, "src")) \
               and os.path.isdir(os.path.join(root, "configs")):
                return root
    return None


def col_pearson(A, B, eps=1e-8):
    """Column-wise Pearson r between two (n_donor x n_gene) matrices."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0)) * np.sqrt((B ** 2).sum(0))
    r = np.full(A.shape[1], np.nan, dtype=np.float64)
    ok = den > eps
    r[ok] = num[ok] / den[ok]
    return r


def summarize(r):
    v = r[np.isfinite(r)]
    if v.size == 0:
        return dict(n_genes=0, median=np.nan, mean=np.nan, f10=np.nan, f30=np.nan, f50=np.nan)
    return dict(n_genes=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)),
                f10=float((v > 0.1).mean()), f30=float((v > 0.3).mean()), f50=float((v > 0.5).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/GTEX_data.csv")
    ap.add_argument("--model", default="data/normalised_model_default.pth")
    ap.add_argument("--out",   default="recoverability")
    ap.add_argument("--source", default="Whole_Blood")
    ap.add_argument("--min-paired", type=int, default=25)
    ap.add_argument("--hyfa-root", default=None, help="path to the HYFA repo (auto-detected if omitted)")
    a = ap.parse_args()

    # ---- locate repo, resolve I/O paths, then switch into the repo --------------
    orig_cwd = os.getcwd()
    hyfa_root = find_hyfa_root(a.hyfa_root)
    if hyfa_root is None:
        sys.exit("Could not find the HYFA repo (a folder with train_gtex.py + src/ + configs/).\n"
                 "Pass --hyfa-root PATH, e.g. --hyfa-root HYFA-main\\HYFA-main")
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(orig_cwd, a.out)
    def resolve_in(p):
        if os.path.isabs(p): return p
        for base in (orig_cwd, hyfa_root):
            cand = os.path.join(base, p)
            if os.path.exists(cand): return cand
        return os.path.join(hyfa_root, p)      # let it error informatively
    data_path, model_path = resolve_in(a.data), resolve_in(a.model)

    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "recoverability_console.txt"), "w")
    def LOG(*m):
        s = " ".join(str(x) for x in m); print(s); logf.write(s + "\n"); logf.flush()
    LOG("HYFA repo   : %s" % hyfa_root)
    LOG("launched in : %s" % orig_cwd)
    LOG("outputs to  : %s\n" % out_dir)

    meta_path = os.path.join(hyfa_root, "data", "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt")
    missing = [p for p in (data_path, model_path, meta_path) if not os.path.exists(p)]
    if missing:
        LOG("ERROR: missing required file(s):")
        for p in missing: LOG("   " + p)
        LOG("\nPut the three README downloads in %s" % os.path.join(hyfa_root, "data"))
        LOG("  GTEX_data.csv | GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt | normalised_model_default.pth")
        sys.exit(1)

    os.chdir(hyfa_root)                          # HYFA's imports + hardcoded paths are CWD-relative
    if hyfa_root not in sys.path: sys.path.insert(0, hyfa_root)

    # ---- verified compat shims (from the prior run) ----------------------------
    import torch, wandb
    def _torch_load_compat(path):
        try:    return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError: return torch.load(path, map_location="cpu")
    import torch_scatter as _ts
    def _long_idx(fn):
        def wrapped(src, index, *args, **kwargs):
            if index.dtype != torch.long: index = index.long()
            return fn(src, index, *args, **kwargs)
        return wrapped
    for _n in ("scatter_max", "scatter_add", "scatter_mean", "scatter_min", "scatter_std", "scatter_softmax"):
        if hasattr(_ts, _n): setattr(_ts, _n, _long_idx(getattr(_ts, _n)))

    # ---- HYFA (imported from repo root; reuses train_gtex's namespace) ----------
    from train_gtex import (GTEx_v8_normalised_adata, HypergraphDataset,
                            HypergraphNeuralNet, Data, DataLoader)
    from src.train_utils import forward

    wandb.init(project="multitissue_imputation", config="configs/default.yaml", mode="disabled")
    config = wandb.config

    LOG("loading authors' GTEx adata ...")
    adata = GTEx_v8_normalised_adata(file=data_path)
    device = torch.device("cpu")
    config.update({"static_node_types": {
        "Tissue": (len(adata.obs["Tissue_idx"].unique()), config.d_tissue),
        "metagenes": (config.meta_G, config.d_gene)}}, allow_val_change=True)
    config.update({"dynamic_node_types": {
        "Participant ID": (len(adata.obs["Participant ID"].unique()), config.d_patient)}},
        allow_val_change=True)
    config.G = adata.shape[-1]
    model = HypergraphNeuralNet(config).to(device)
    model.load_state_dict(_torch_load_compat(model_path))
    model.eval()
    LOG("pretrained HYFA loaded.\n")

    tissues_all = list(adata.obs["Tissue"].unique())
    if a.source not in tissues_all:
        LOG("ERROR: source tissue '%s' not in data. Available:" % a.source)
        LOG("  " + ", ".join(sorted(tissues_all))); sys.exit(1)

    donor_tissues = adata.obs.groupby("Participant ID")["Tissue"].agg(set).to_dict()
    targets = [t for t in tissues_all if t != a.source]
    symbols = np.asarray(adata.var["Symbol"].values)

    brain_like  = [t for t in targets if "brain" in t.lower()]
    cortex_like = [t for t in brain_like if ("cortex" in t.lower() or "frontal" in t.lower())]
    LOG("source: %s | %d target tissues | brain: %d | cortex-like: %s\n"
        % (a.source, len(targets), len(brain_like), ", ".join(cortex_like) or "<none>"))

    rows = []
    for tt in sorted(targets):
        paired = [d for d, ts in donor_tissues.items() if a.source in ts and tt in ts]
        if len(paired) < a.min_paired:
            continue
        try:
            ds = HypergraphDataset(adata,
                                   obs_source={"Tissue": [a.source], "Participant ID": paired},
                                   obs_target={"Tissue": [tt], "Participant ID": paired},
                                   static=True)
            loader = DataLoader(ds, batch_size=len(ds), collate_fn=Data.from_datalist, shuffle=False)
            d = next(iter(loader))
            with torch.no_grad():
                out, _ = forward(d, model, device, preprocess_fn=None)
                pred = out["px_rate"].cpu().numpy()
                true = d.x_target.cpu().numpy()
        except Exception as e:
            LOG("  %-28s SKIP (%s)" % (tt, e)); continue

        r = col_pearson(true, pred)
        s = summarize(r); s.update(tissue=tt, n_paired=len(paired))
        rows.append(s)
        LOG("  %-28s n=%4d  median_r=%6.3f  frac>0.3=%5.1f%%  frac>0.1=%5.1f%%"
            % (tt, len(paired), s["median"], 100 * s["f30"], 100 * s["f10"]))

        if tt in brain_like:                       # dump per-gene r for later module intersection
            order = np.argsort(-np.nan_to_num(r, nan=-np.inf))
            tsd, psd = true.std(0), pred.std(0)
            with open(os.path.join(out_dir, "pergene_%s.csv" % tt), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["gene", "pearson_r", "true_sd", "pred_sd"])
                for j in order:
                    w.writerow([symbols[j],
                                "" if not np.isfinite(r[j]) else round(float(r[j]), 4),
                                round(float(tsd[j]), 4), round(float(psd[j]), 4)])

    if not rows:
        LOG("\nNo tissue met --min-paired=%d with source %s." % (a.min_paired, a.source)); sys.exit(1)

    rows_sorted = sorted(rows, key=lambda s: (s["median"] if np.isfinite(s["median"]) else 1e9))
    with open(os.path.join(out_dir, "recoverability_by_tissue.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tissue", "n_paired", "n_genes", "median_r", "mean_r",
                    "frac_r_gt_0.1", "frac_r_gt_0.3", "frac_r_gt_0.5"])
        for s in rows_sorted:
            w.writerow([s["tissue"], s["n_paired"], s["n_genes"], round(s["median"], 4),
                        round(s["mean"], 4), round(s["f10"], 4), round(s["f30"], 4), round(s["f50"], 4)])

    n = len(rows_sorted)
    LOG("\n" + "=" * 74)
    LOG("PER-GENE RECOVERABILITY FROM BLOOD  (median r, worst -> best)")
    LOG("=" * 74)
    for rank, s in enumerate(rows_sorted, 1):
        star = "   <-- brain" if s["tissue"] in brain_like else ""
        LOG("  %2d/%d  %-28s median_r=%6.3f  frac>0.3=%5.1f%%%s"
            % (rank, n, s["tissue"], s["median"], 100 * s["f30"], star))

    LOG("\n" + "-" * 74)
    verdict_targets = cortex_like if cortex_like else brain_like
    verdict_json = {"source": a.source, "min_paired": a.min_paired, "n_tissues": n,
                    "note": "optimistic ceiling: paired donors include GTEx training donors",
                    "cortex_targets": []}
    if not verdict_targets:
        LOG("No brain/cortex tissue in GTEx labels; inspect the ranking above.")
    for tt in verdict_targets:
        s = next(x for x in rows_sorted if x["tissue"] == tt)
        rank = rows_sorted.index(s) + 1
        gate = bool(s["f30"] >= 0.10)
        LOG("CORTEX GATE  [%s]%s" % (tt, "   (BA9 = the HD tissue leg)" if "frontal" in tt.lower() else ""))
        LOG("  median per-gene r = %.3f | recoverable: %.1f%% (r>0.3), %.1f%% (r>0.1) | rank %d/%d (1=worst)"
            % (s["median"], 100 * s["f30"], 100 * s["f10"], rank, n))
        if not gate:
            LOG("  -> only %.1f%% of cortex genes carry recoverable donor-level signal from blood." % (100 * s["f30"]))
            LOG("     The HD module must land in that minority for imputation (and any GRN lift) to")
            LOG("     have material. After 03/04, intersect the module with pergene_%s.csv BEFORE" % tt)
            LOG("     committing to the HD imputation leg.")
        else:
            LOG("  -> %.1f%% of cortex genes recoverable (r>0.3): real headroom exists. Proceed;" % (100 * s["f30"]))
            LOG("     confirm at 05 that the HD module falls in the recoverable set.")
        verdict_json["cortex_targets"].append(
            {"tissue": tt, "median_r": round(s["median"], 4), "frac_gt_0.3": round(s["f30"], 4),
             "frac_gt_0.1": round(s["f10"], 4), "rank_worst_first": rank, "recoverable_gate": gate})
    LOG("-" * 74)

    with open(os.path.join(out_dir, "gate_verdict.json"), "w") as fh:
        json.dump(verdict_json, fh, indent=2)
    LOG("outputs in: %s" % out_dir)
    logf.close()


if __name__ == "__main__":
    main()
