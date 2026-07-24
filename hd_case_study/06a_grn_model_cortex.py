#!/usr/bin/env python3
"""
06a_grn_model_cortex.py -- GRN model on real HD-vs-control cortex.
Fits per-class Gaussian conditionals over the DEG-GRN LCC edges and classifies HD/control:
expression vs expression+GRN (edge and coarse). --imputable-only restricts to the
HYFA-imputable induced subgraph (the 74-gene object used downstream). Clean-data control
matched to the imputed test in 06b.

Inputs : prep/hd_cortex_ba9_counts_by_symbol.csv, prep/hd_cortex_ba9_pheno.csv,
         grn/grn_lcc_edges.csv   (--imputable-only also needs grn/gene_coverage.csv)
Outputs: grn/grn_model_cortex[_imputable]/  (summary + conditionals_full.csv)
Usage  : python 06a_grn_model_cortex.py [--imputable-only] [--folds 5] [--seed 0] [--shrink 2.0]
"""
import argparse, csv, os, sys
import numpy as np
import pandas as pd

LOG2PI = np.log(2 * np.pi)


def find_root():
    f = globals().get("__file__")
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def auc_rank(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    u = ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def fit_line(x, y):
    if len(x) < 3 or np.std(x) < 1e-9:
        return (float(np.mean(y)) if len(y) else 0.0), 0.0, (float(np.var(y)) if len(y) else 1.0)
    b, a = np.polyfit(x, y, 1)
    return a, b, float(np.mean((y - (a + b * x)) ** 2))


def fit_conditionals(E, exprm, ridx, ctrl_cols, hd_cols, shrink, floor=1e-3):
    P = np.zeros((len(E), 6))
    both = np.concatenate([ctrl_cols, hd_cols])
    for k, (s, t) in enumerate(E):
        sr, tr = ridx[s], ridx[t]
        marg = float(np.var(exprm[tr, both])) + floor
        a_c, b_c, m_c = fit_line(exprm[sr, ctrl_cols], exprm[tr, ctrl_cols])
        a_h, b_h, m_h = fit_line(exprm[sr, hd_cols], exprm[tr, hd_cols])
        nc, nh = len(ctrl_cols), len(hd_cols)
        s2_c = max((nc * m_c + shrink * marg) / (nc + shrink), floor)
        s2_h = max((nh * m_h + shrink * marg) / (nh + shrink), floor)
        P[k] = (a_c, b_c, s2_c, a_h, b_h, s2_h)
    return P


def llr_features(E, exprm, ridx, P, cols):
    X = np.zeros((len(cols), len(E)))
    for k, (s, t) in enumerate(E):
        sr, tr = ridx[s], ridx[t]
        si = exprm[sr, cols]; ti = exprm[tr, cols]
        a_c, b_c, s2_c, a_h, b_h, s2_h = P[k]
        X[:, k] = (-0.5 * (LOG2PI + np.log(s2_h)) - (ti - (a_h + b_h * si)) ** 2 / (2 * s2_h)) \
                  - (-0.5 * (LOG2PI + np.log(s2_c)) - (ti - (a_c + b_c * si)) ** 2 / (2 * s2_c))
    return X


def coarse_from_edge(Xedge, comm_edge_idx):
    return np.column_stack([Xedge[:, idx].sum(1) if idx else np.zeros(Xedge.shape[0])
                            for idx in comm_edge_idx]) if comm_edge_idx else np.zeros((Xedge.shape[0], 0))


def cv_auc(E, exprm, ridx, y, expr_rows, comm_edge_idx, folds, seed, shrink, level):
    """
    Leakage-safe CV-AUC of [expression block] (+ optional inferred-edge block).
    level: 'none' (expression only), 'edge', or 'coarse'.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    n = len(y); cols = np.arange(n)
    exprX = exprm[np.ix_(expr_rows, cols)].T          # samples x module-genes (label-free)
    proba = np.zeros(n)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(cols, y):
        blocks_tr = [exprX[tr]]; blocks_te = [exprX[te]]
        if level != "none":
            ctrl_tr = cols[tr][y[tr] == 0]; hd_tr = cols[tr][y[tr] == 1]
            if len(ctrl_tr) < 3 or len(hd_tr) < 3:
                proba[te] = 0.5; continue
            P = fit_conditionals(E, exprm, ridx, ctrl_tr, hd_tr, shrink)
            Etr = llr_features(E, exprm, ridx, P, cols[tr])
            Ete = llr_features(E, exprm, ridx, P, cols[te])
            if level == "coarse":
                Etr = coarse_from_edge(Etr, comm_edge_idx); Ete = coarse_from_edge(Ete, comm_edge_idx)
            blocks_tr.append(Etr); blocks_te.append(Ete)
        Xtr = np.hstack(blocks_tr); Xte = np.hstack(blocks_te)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), y[tr])
        proba[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return auc_rank(proba, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink", type=float, default=2.0)
    ap.add_argument("--imputable-only", action="store_true",
                    help="restrict to HYFA-imputable genes (in_hyfa_panel) + edges among them")
    ap.add_argument("--cov", default=None, help="gene_coverage.csv (default grn/gene_coverage.csv)")
    a = ap.parse_args()

    root = find_root()
    prep = os.path.join(root, "prep"); grn = os.path.join(root, "grn")
    out = os.path.join(grn, "grn_model_cortex_imputable" if a.imputable_only else "grn_model_cortex")
    os.makedirs(out, exist_ok=True)
    counts_f = os.path.join(prep, "hd_cortex_ba9_counts_by_symbol.csv")
    pheno_f = os.path.join(prep, "hd_cortex_ba9_pheno.csv")
    edges_f = os.path.join(grn, "grn_lcc_edges.csv")
    for f in (counts_f, pheno_f, edges_f):
        if not os.path.exists(f):
            sys.exit("missing input (run 02 and 05 first): " + f)

    cnt = pd.read_csv(counts_f); cnt = cnt.set_index(cnt.columns[0])
    lib = cnt.sum(0).replace(0, np.nan)
    logcpm = np.log2(cnt.divide(lib, axis=1) * 1e6 + 1.0)
    ph = pd.read_csv(pheno_f).set_index("sample_id").reindex(logcpm.columns)
    keep = ph["condition"].isin(["HD", "Control"]).values
    logcpm = logcpm.loc[:, keep]; ph = ph.loc[keep]
    y = (ph["condition"].values == "HD").astype(int)
    print("cortex: HD=%d Control=%d | genes=%d" % (y.sum(), (1 - y).sum(), logcpm.shape[0]))

    ed = pd.read_csv(edges_f); present = set(logcpm.index)
    edges = [(s, t) for s, t in zip(ed["source"], ed["target"])
             if s in present and t in present and s != t]

    if a.imputable_only:                       # keep HYFA-imputable genes + induced subgraph
        cov_f = a.cov or os.path.join(grn, "gene_coverage.csv")
        if not os.path.exists(cov_f):
            sys.exit("--imputable-only needs %s (run 06c first)" % cov_f)
        cov = pd.read_csv(cov_f)
        keep_g = set(g for g, ok in zip(cov["gene"], cov["in_hyfa_panel"]) if ok)
        n_before = len(edges)
        edges = [(s, t) for s, t in edges if s in keep_g and t in keep_g]
        print("[imputable-only] restricted to induced subgraph: %d -> %d edges" % (n_before, len(edges)))

    genes_used = sorted({g for e in edges for g in e})
    if len(edges) < 10:
        sys.exit("too few usable edges; check symbol overlap")

    import networkx as nx
    U = nx.Graph(); U.add_edges_from(edges)
    try:
        comms = [c for c in nx.algorithms.community.greedy_modularity_communities(U) if len(c) >= 3]
    except Exception:
        comms = [set(c) for c in nx.connected_components(U) if len(c) >= 3]
    comm_edge_idx = [[k for k, (s, t) in enumerate(edges) if s in c and t in c] for c in comms]
    print("LCC edges=%d over %d genes | communities(>=3)=%d" % (len(edges), len(genes_used), len(comms)))

    exprm = logcpm.values.astype(float)
    ridx = {g: i for i, g in enumerate(logcpm.index)}
    expr_rows = np.array(sorted(ridx[g] for g in genes_used))

    auc_expr = cv_auc(edges, exprm, ridx, y, expr_rows, comm_edge_idx, a.folds, a.seed, a.shrink, "none")
    auc_edge = cv_auc(edges, exprm, ridx, y, expr_rows, comm_edge_idx, a.folds, a.seed, a.shrink, "edge")
    auc_coarse = cv_auc(edges, exprm, ridx, y, expr_rows, comm_edge_idx, a.folds, a.seed, a.shrink, "coarse")
    print("\nexpression-only        CV-AUC = %.3f" % auc_expr)
    print("expression + real edge CV-AUC = %.3f  (increment %+.3f)" % (auc_edge, auc_edge - auc_expr))
    print("expression + real coarse      = %.3f  (increment %+.3f)" % (auc_coarse, auc_coarse - auc_expr))

    Pfull = fit_conditionals(edges, exprm, ridx, np.where(y == 0)[0], np.where(y == 1)[0], a.shrink)
    with open(os.path.join(out, "conditionals_full.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "a_ctrl", "b_ctrl", "sig2_ctrl", "a_hd", "b_hd", "sig2_hd"])
        for (s, t), p in zip(edges, Pfull):
            w.writerow([s, t] + [round(float(v), 6) for v in p])

    lines = [
        "GRN model: expression vs expression+GRN (real HD-vs-control cortex)",
        "LCC induced subgraph: %d edges, %d genes | HD=%d Control=%d | communities(>=3)=%d"
        % (len(edges), len(genes_used), int(y.sum()), int((1 - y).sum()), len(comms)),
        "",
        "expression only          CV-AUC = %.3f" % auc_expr,
        "expression + GRN edge    CV-AUC = %.3f   increment %+.3f" % (auc_edge, auc_edge - auc_expr),
        "expression + GRN coarse  CV-AUC = %.3f   increment %+.3f" % (auc_coarse, auc_coarse - auc_expr),
        "",
        "(real cortex; expression is saturated by the DEGs, so increments here are near 0 --",
        " this is the clean-data control matched to the imputed test in 06b.)",
    ]
    with open(os.path.join(out, "grn_grn_model_cortex_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("\noutputs in:", out)


if __name__ == "__main__":
    main()
