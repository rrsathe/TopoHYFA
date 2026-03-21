"""
Downstream disease prediction using imputed gene expression.

Python port of the core lasso + AUC loop from Imputation/code/disease_prediction.r.
Compares disease-classification AUC using:
  1) HYFA-imputed Heart expression
  2) Ground-truth Heart expression
  3) Blood-surrogate (raw Whole Blood expression used directly)

Default phenotype: MHHTN (hypertension).
"""

import argparse
import os
from contextlib import suppress

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "results"
METADATA_FILE = "data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"


def run_disease_cv(X, y, n_folds=5, n_ensemble=10):
    """
    Cross-validated lasso disease prediction, averaged over `n_ensemble` runs.
    Returns mean AUC across all folds and ensembles.
    """
    aucs = []
    for _ in range(n_ensemble):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip folds with only one class
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = LassoCV(cv=4, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            y_score = model.predict(X_test)

            with suppress(ValueError):
                aucs.append(roc_auc_score(y_test, y_score))

    return np.mean(aucs) if aucs else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Disease prediction from imputed expression")
    parser.add_argument(
        "--phenotype", default="SEX", help="Phenotype column: SEX or DTHHRDY (default: SEX)"
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-ensemble", type=int, default=10)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    print("Loading predictions and metadata ...")
    pred_df = pd.read_csv(f"{RESULTS_DIR}/hyfa_predictions_test.csv", index_col=0)
    truth_df = pd.read_csv(f"{RESULTS_DIR}/ground_truth_test.csv", index_col=0)

    # Load arrays for blood-surrogate expression
    data = np.load(f"{RESULTS_DIR}/eval_arrays.npz", allow_pickle=True)
    x_test_source = data["x_test_source"]  # blood expression
    gene_symbols = data["gene_symbols"]

    # Build blood surrogate DF with same index as pred_df
    blood_df = pd.DataFrame(x_test_source, columns=gene_symbols, index=pred_df.index)

    # Load phenotype metadata
    pheno = pd.read_csv(METADATA_FILE, delimiter="\t").set_index("SUBJID")

    # ── Build labels ─────────────────────────────────────────────────
    common_ids = pred_df.index[pred_df.index.isin(pheno.index)]

    if len(common_ids) == 0:
        print("WARNING: No overlapping participant IDs found.")
        print(f"  Prediction IDs sample: {pred_df.index[:3].tolist()}")
        print(f"  Phenotype IDs sample:  {pheno.index[:3].tolist()}")
        return

    print(f"Common participants: {len(common_ids)}")

    if args.phenotype not in pheno.columns:
        print(f"ERROR: Phenotype '{args.phenotype}' not found. Available: {list(pheno.columns)}")
        return

    raw_labels = pheno.loc[common_ids, args.phenotype].values

    # Encode labels as binary
    if args.phenotype == "SEX":
        # GTEx: 1=Male, 2=Female -> 0=Male, 1=Female
        raw_numeric = pd.to_numeric(pd.Series(raw_labels), errors="coerce").to_numpy(dtype=float)
        valid = (~np.isnan(raw_numeric)) & ((raw_numeric == 1.0) | (raw_numeric == 2.0))
        labels = np.zeros(raw_numeric.shape[0], dtype=int)
        labels[valid] = (raw_numeric[valid] - 1.0).astype(int)
    elif args.phenotype == "DTHHRDY":
        # Hardy scale: 0=Ventilator, 1=Violent/fast, 2=Fast natural, 3=Intermediate, 4=Slow
        # Binarize: 0 (ventilator) vs 1-4 (non-ventilator)
        raw_numeric = pd.to_numeric(pd.Series(raw_labels), errors="coerce").to_numpy(dtype=float)
        labels = (raw_numeric > 0).astype(int)
        valid = ~np.isnan(raw_numeric)
    else:
        labels = raw_labels.astype(float)
        valid = (labels == 0) | (labels == 1)
        labels = labels.astype(int)

    common_ids = common_ids[valid]
    labels = np.asarray(labels[valid], dtype=int)

    n_cases = int(np.count_nonzero(labels == 1))
    n_controls = int(np.count_nonzero(labels == 0))
    print(f"Phenotype: {args.phenotype}  |  Cases: {n_cases}  |  Controls: {n_controls}")

    if n_cases < 5 or n_controls < 5:
        print("Too few cases or controls for meaningful CV. Aborting.")
        return

    X_hyfa = pred_df.loc[common_ids].values
    X_truth = truth_df.loc[common_ids].values
    X_blood = blood_df.loc[common_ids].values

    # ── Run disease prediction ───────────────────────────────────────
    print(f"\nRunning {args.n_folds}-fold CV x {args.n_ensemble} ensembles ...")

    auc_hyfa = run_disease_cv(X_hyfa, labels, args.n_folds, args.n_ensemble)
    print(f"  HYFA imputed AUC:    {auc_hyfa:.4f}")

    auc_truth = run_disease_cv(X_truth, labels, args.n_folds, args.n_ensemble)
    print(f"  Ground truth AUC:    {auc_truth:.4f}")

    auc_blood = run_disease_cv(X_blood, labels, args.n_folds, args.n_ensemble)
    print(f"  Blood surrogate AUC: {auc_blood:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    results = pd.DataFrame(
        {
            "Method": ["HYFA (imputed)", "Ground truth (Heart)", "Blood surrogate"],
            "AUC": [auc_hyfa, auc_truth, auc_blood],
            "Phenotype": [args.phenotype] * 3,
            "N_cases": [n_cases] * 3,
            "N_controls": [n_controls] * 3,
        }
    )
    print(f"\n{results.to_string(index=False)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(f"{RESULTS_DIR}/disease_prediction_results.csv", index=False)
    print(f"\nSaved -> {RESULTS_DIR}/disease_prediction_results.csv")


if __name__ == "__main__":
    main()
