#!/usr/bin/env python3
"""
06b_classify.py -- does the GRN improve HD/control classification of the imputed BA9 cortex?
On the object = 138-node LCC restricted to HYFA-imputable genes (74-gene / 102-edge induced
subgraph), per contrast (manifest, carrier): expression only vs expression+GRN edge vs
expression+GRN coarse, with the increment. Same gene set in every arm; edges are the only
difference. Conditionals fit (frozen) on inverse-normal real cortex, applied to inverse-normal
imputed cortex; cortex and blood are separate cohorts, so no circularity.

Inputs : prep/hd_blood_imputed_ba9.csv, prep/hd_cortex_ba9_counts_by_symbol.csv,
         prep/hd_cortex_ba9_pheno.csv, prep/hd_blood_labels.csv,
         grn/grn_lcc_edges.csv, grn/gene_coverage.csv
Outputs: grn/classify/  (classify_summary.txt, classify_results.csv)
Usage  : python 06b_classify.py [--folds 5] [--seed 0] [--shrink 2.0]
"""
import argparse, os, sys
import numpy as np
import pandas as pd

LOG2PI = np.log(2 * np.pi)


def find_root():
    f = globals().get("__file__")
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def inverse_normal_rows(M):
    from scipy.stats import rankdata, norm
    out = np.zeros_like(M, float); n = M.shape[1]
    for i in range(M.shape[0]):
        r = rankdata(M[i, :], method="average")
        out[i, :] = norm.ppf((r - 0.375) / (n + 0.25))
    return out


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
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def cv_predict(X, y, folds, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    if X.shape[1] == 0:
        return np.full(len(y), 0.5)
    proba = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr]); proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return proba


def fit_line(x, y):
    if len(x) < 3 or np.std(x) < 1e-9:
        return (float(np.mean(y)) if len(y) else 0.0), 0.0, (float(np.var(y)) if len(y) else 1.0)
    b, a = np.polyfit(x, y, 1)
    return a, b, float(np.mean((y - (a + b * x)) ** 2))


def fit_conditionals(E, exprm, ridx, ctrl_cols, hd_cols, shrink, floor=1e-3):
    P = np.zeros((len(E), 6)); both = np.concatenate([ctrl_cols, hd_cols])
    for k, (s, t) in enumerate(E):
        sr, tr = ridx[s], ridx[t]
        marg = float(np.var(exprm[tr, both])) + floor
        a_c, b_c, m_c = fit_line(exprm[sr, ctrl_cols], exprm[tr, ctrl_cols])
        a_h, b_h, m_h = fit_line(exprm[sr, hd_cols], exprm[tr, hd_cols])
        nc, nh = len(ctrl_cols), len(hd_cols)
        P[k] = (a_c, b_c, max((nc * m_c + shrink * marg) / (nc + shrink), floor),
                a_h, b_h, max((nh * m_h + shrink * marg) / (nh + shrink), floor))
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


def coarse_from_edge(Xe, cei):
    return np.column_stack([Xe[:, idx].sum(1) if idx else np.zeros(Xe.shape[0]) for idx in cei]) \
        if cei else np.zeros((Xe.shape[0], 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink", type=float, default=2.0)
    a = ap.parse_args()

    root = find_root(); prep = os.path.join(root, "prep"); grn = os.path.join(root, "grn")
    out = os.path.join(grn, "classify"); os.makedirs(out, exist_ok=True)
    imp_f = os.path.join(prep, "hd_blood_imputed_ba9.csv")
    cortex_f = os.path.join(prep, "hd_cortex_ba9_counts_by_symbol.csv")
    cph_f = os.path.join(prep, "hd_cortex_ba9_pheno.csv")
    labels_f = os.path.join(prep, "hd_blood_labels.csv")
    edges_f = os.path.join(grn, "grn_lcc_edges.csv")
    cov_f = os.path.join(grn, "gene_coverage.csv")
    for f in (imp_f, cortex_f, cph_f, labels_f, edges_f, cov_f):
        if not os.path.exists(f):
            sys.exit("missing input: " + f)

    imp = pd.read_csv(imp_f).set_index("sample_id"); imp_samples = list(imp.index)
    imp_mat = inverse_normal_rows(imp.values.T.astype(float)); ridx_imp = {g: i for i, g in enumerate(imp.columns)}
    cnt = pd.read_csv(cortex_f); cnt = cnt.set_index(cnt.columns[0])
    lib = cnt.sum(0).replace(0, np.nan)
    cortex = inverse_normal_rows((cnt.divide(lib, axis=1) * 1e6).values); ridx_cx = {g: i for i, g in enumerate(cnt.index)}
    cph = pd.read_csv(cph_f).set_index("sample_id").reindex(cnt.columns)
    cy = np.where(cph["condition"].values == "HD", 1, np.where(cph["condition"].values == "Control", 0, -1))
    ctrl_cx = np.where(cy == 0)[0]; hd_cx = np.where(cy == 1)[0]

    # object = 138-LCC restricted to HYFA-imputable genes -> induced subgraph (74 genes, 102 edges)
    cov = pd.read_csv(cov_f)
    Iset = set(g for g, ok in zip(cov["gene"], cov["in_hyfa_panel"]) if ok and g in ridx_cx and g in ridx_imp)
    ed = pd.read_csv(edges_f)
    edges = [(s, t) for s, t in zip(ed["source"], ed["target"]) if s in Iset and t in Iset and s != t]
    genes = sorted({g for e in edges for g in e})              # the 74 induced-subgraph genes
    print("induced subgraph: %d edges over %d genes (LCC ∩ HYFA-imputable)" % (len(edges), len(genes)))

    import networkx as nx
    U = nx.Graph(); U.add_edges_from(edges)
    try:
        comms = [c for c in nx.algorithms.community.greedy_modularity_communities(U) if len(c) >= 3]
    except Exception:
        comms = [set(c) for c in nx.connected_components(U) if len(c) >= 3]
    cei = [[k for k, (s, t) in enumerate(edges) if s in c and t in c] for c in comms]

    def rows_imp(g): return np.array([ridx_imp[x] for x in g]) if g else np.array([], int)
    exprX = imp_mat[rows_imp(genes)].T                          # 124 x 74 (the induced-subgraph genes)
    P_real = fit_conditionals(edges, cortex, ridx_cx, ctrl_cx, hd_cx, a.shrink)   # frozen on real cortex
    edgeX = llr_features(edges, imp_mat, ridx_imp, P_real, np.arange(len(imp_samples)))
    coarseX = coarse_from_edge(edgeX, cei)

    lab = pd.read_csv(labels_f).set_index("sample_id").reindex(imp_samples)
    rows = []; pred_rows = []; report = [
        "06b v3: expression vs expression+GRN on imputed BA9 (induced subgraph)",
        "genes=%d | edges=%d | communities(>=3)=%d | folds=%d" % (len(genes), len(edges), len(comms), a.folds), ""]

    for contrast, col in (("manifest", "contrast_manifest"), ("carrier", "contrast_carrier")):
        lc = lab[col].astype(str); mask = lc.isin(["HD", "Control"]).values
        idx = np.where(mask)[0]; y = (lc.values[mask] == "HD").astype(int)
        if y.sum() < 5 or (1 - y).sum() < 5:
            report.append("[%s] skipped" % contrast); continue
        ex = exprX[idx]; edr = edgeX[idx]; cor = coarseX[idx]
        p_e0 = cv_predict(ex, y, a.folds, a.seed)
        p_ed = cv_predict(np.hstack([ex, edr]), y, a.folds, a.seed)
        p_co = cv_predict(np.hstack([ex, cor]), y, a.folds, a.seed)
        auc_e0, auc_ed, auc_co = auc_rank(p_e0, y), auc_rank(p_ed, y), auc_rank(p_co, y)
        report += [
            "===== %s   (HD=%d, Control=%d) =====" % (contrast, int(y.sum()), int((1 - y).sum())),
            "  expression only          AUC = %.3f" % auc_e0,
            "  expression + GRN edge    AUC = %.3f   increment %+.3f" % (auc_ed, auc_ed - auc_e0),
            "  expression + GRN coarse  AUC = %.3f   increment %+.3f" % (auc_co, auc_co - auc_e0),
            ""]
        rows.append(dict(contrast=contrast, n_hd=int(y.sum()), n_ctrl=int((1 - y).sum()),
                         auc_expr=round(auc_e0, 4), auc_expr_edge=round(auc_ed, 4),
                         auc_expr_coarse=round(auc_co, 4),
                         increment_edge=round(auc_ed - auc_e0, 4), increment_coarse=round(auc_co - auc_e0, 4)))
        sids = np.array(imp_samples)[idx]
        for sid, yy, a0, a1, a2 in zip(sids, y, p_e0, p_ed, p_co):
            pred_rows.append(dict(contrast=contrast, sample_id=sid, y_true=int(yy),
                                  p_expr=round(float(a0), 5), p_expr_edge=round(float(a1), 5),
                                  p_expr_coarse=round(float(a2), 5)))

    with open(os.path.join(out, "classify_summary.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(out, "classify_results.csv"), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(out, "predictions.csv"), index=False)   # for ROC (Fig 4B)
    print("\n" + "\n".join(report)); print("outputs in:", out)


if __name__ == "__main__":
    main()
