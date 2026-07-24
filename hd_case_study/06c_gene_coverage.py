#!/usr/bin/env python3
"""
06c_gene_coverage.py -- audit where module genes are lost between the GRN and the imputed test.
Per module gene, flags presence in the real cortex matrix, HYFA's imputable panel, and the raw
blood matrix, and cross-references dropped genes against the significant cortex DEGs.

Inputs : grn/grn_lcc_nodes.txt, prep/hd_cortex_ba9_counts_by_symbol.csv,
         prep/hd_blood_imputed_ba9.csv, prep/hd_blood_counts_by_symbol.csv, deg/deg_cortex_significant.csv
Output : grn/gene_coverage.csv + printed summary
Usage  : python 06c_gene_coverage.py
"""
import os, sys
import pandas as pd


def find_root():
    f = globals().get("__file__")
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def col1(path):
    return set(pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str))


def main():
    root = find_root(); prep = os.path.join(root, "prep"); grn = os.path.join(root, "grn")
    nodes_f = os.path.join(grn, "grn_lcc_nodes.txt")
    cortex_f = os.path.join(prep, "hd_cortex_ba9_counts_by_symbol.csv")
    imp_f = os.path.join(prep, "hd_blood_imputed_ba9.csv")
    blood_f = os.path.join(prep, "hd_blood_counts_by_symbol.csv")
    deg_f = os.path.join(root, "deg", "deg_cortex_significant.csv")
    for f in (nodes_f, cortex_f, imp_f, blood_f):
        if not os.path.exists(f):
            sys.exit("missing input: " + f)

    module = [l.strip() for l in open(nodes_f) if l.strip()]
    cortex_genes = col1(cortex_f)
    blood_genes = col1(blood_f)
    panel = set(c for c in pd.read_csv(imp_f, nrows=1).columns if c != "sample_id")

    deg = None
    if os.path.exists(deg_f):
        deg = pd.read_csv(deg_f)
        gcol = "gene_symbol" if "gene_symbol" in deg.columns else deg.columns[0]
        lcol = "logFC" if "logFC" in deg.columns else None
        deg = deg.set_index(gcol)

    rows = []
    for g in module:
        r = {"gene": g, "in_cortex": g in cortex_genes, "in_hyfa_panel": g in panel,
             "in_blood_raw": g in blood_genes,
             "is_sig_deg": bool(deg is not None and g in deg.index)}
        if deg is not None and g in deg.index and lcol:
            r["deg_logFC"] = round(float(deg.loc[g, lcol]), 3)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(grn, "gene_coverage.csv"), index=False)

    n = len(module)
    inpanel = df["in_hyfa_panel"].sum()
    dropped = df[~df["in_hyfa_panel"]]
    print("module genes: %d" % n)
    print("  in real cortex   : %d  (expect all -- they came from cortex DEGs)" % df["in_cortex"].sum())
    print("  in HYFA panel    : %d  <- imputable" % inpanel)
    print("  in raw blood     : %d" % df["in_blood_raw"].sum())
    print("  DROPPED (not in HYFA panel): %d" % len(dropped))
    if deg is not None:
        print("  of the dropped, significant DEGs: %d / %d" % (int(dropped["is_sig_deg"].sum()), len(dropped)))
        if "deg_logFC" in dropped.columns:
            top = dropped.dropna(subset=["deg_logFC"]).reindex(
                dropped["deg_logFC"].abs().sort_values(ascending=False).index).head(12)
            if len(top):
                print("  strongest dropped DEGs (|logFC|):")
                for _, x in top.iterrows():
                    print("     %-12s logFC=%+.2f" % (x["gene"], x["deg_logFC"]))
    print("\n-> grn/gene_coverage.csv")
    print("If many strong DEGs are in the dropped set, the imputed test ran on a weakened module,")
    print("and HYFA's 12,557-gene panel -- not the method -- is capping coverage.")


if __name__ == "__main__":
    main()
