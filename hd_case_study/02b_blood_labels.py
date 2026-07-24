#!/usr/bin/env python3
"""
02b_blood_labels.py   [run order: 02b]

Derive clean HD-blood labels from the SRP032279 phenotype (02 already saved the
attribute columns; no re-download). SRP032279 encodes cases as mutation CARRIERS,
not "Huntington", which is why 02's generic parser left 91 NAs.

carrier vs control  : attr_mutation_carrier_status  (carrier=91, control=33)
manifest split      : carriers with TFC score < 13 are functionally affected
                      (manifest); carriers at TFC 13 are premanifest.
                      Controls are non-carriers (TFC 11-13, CAG < 36).

Writes prep/hd_blood_labels.csv keyed by sample_id (matches the counts columns):
  sample_id, carrier_status, cag, tfc, stage, manifest_tfc, manifest_stage,
  contrast_carrier   (carrier->HD / control->Control),
  contrast_manifest  (manifest carrier->HD / control->Control / premanifest->NA)

RECORD: if it errors, fix and re-run.
Usage: python 02b_blood_labels.py
"""
import glob, os, sys
import numpy as np
import pandas as pd


def find_root():
    f = globals().get("__file__")
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def main():
    root = find_root()
    cands = ([os.path.join(root, "prep", "hd_blood_pheno_full.csv")]
             + glob.glob(os.path.join(root, "**", "hd_blood_pheno_full.csv"), recursive=True))
    src = next((p for p in cands if os.path.exists(p)), None)
    if src is None:
        sys.exit("hd_blood_pheno_full.csv not found under %s (run 02 first)" % root)
    out = os.path.join(os.path.dirname(src), "hd_blood_labels.csv")

    df = pd.read_csv(src)
    if "sample_id" not in df.columns:
        sys.exit("no sample_id column in " + src)
    status = df["attr_mutation_carrier_status"].astype(str).str.strip().str.lower()
    cag   = pd.to_numeric(df.get("attr_cag_repeats"), errors="coerce")
    tfc   = pd.to_numeric(df.get("attr_tfc_score"), errors="coerce")
    stage = pd.to_numeric(df.get("attr_tfc_disease_stage"), errors="coerce")

    is_carrier = status.eq("carrier")
    is_control = status.eq("control")
    manifest_tfc   = is_carrier & (tfc < 13)      # any functional decline
    manifest_stage = is_carrier & (stage >= 2)    # stricter alternative

    contrast_carrier = np.where(is_carrier, "HD", np.where(is_control, "Control", ""))
    contrast_manifest = np.where(manifest_tfc, "HD",
                         np.where(is_control, "Control", ""))  # premanifest carriers -> blank

    res = pd.DataFrame({
        "sample_id": df["sample_id"], "carrier_status": status,
        "cag": cag, "tfc": tfc, "stage": stage,
        "manifest_tfc": manifest_tfc, "manifest_stage": manifest_stage,
        "contrast_carrier": contrast_carrier, "contrast_manifest": contrast_manifest,
    })
    res.to_csv(out, index=False)

    def n(mask): return int(np.sum(mask))
    print("wrote", out)
    print("  carrier vs control      : HD=%d  Control=%d" % (n(is_carrier), n(is_control)))
    print("  manifest(TFC<13) vs ctrl: HD=%d  Control=%d  (premanifest held out=%d)"
          % (n(manifest_tfc), n(is_control), n(is_carrier & (tfc == 13))))
    print("  manifest(stage>=2) alt  : HD=%d" % n(manifest_stage))


if __name__ == "__main__":
    main()
