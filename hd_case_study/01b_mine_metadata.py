#!/usr/bin/env python3
"""
01b_mine_metadata.py   [run order: 01b]

Mine the recount3 web export (study titles + abstracts, which available_projects()
does NOT give us) to find, for a disease application:
  - WHOLE-BLOOD case/control cohorts  (the blood leg; must match GTEx 'Whole Blood')
  - disease-vs-normal TISSUE cohorts  (the tissue leg; DEG/DNB + imputation target)
cross-referenced with GTEx (32) and TCGA (33) tissue availability, then reports the
diseases that have BOTH legs in-snapshot.

Input : the recount3 web export CSV (recount3_selection_*.csv). Located dynamically:
        pass a path as arg1, else the script globs for it under the project root.
Output: <root>/survey/
          catalog_gtex_tissues.csv, catalog_tcga_projects.csv
          blood_cohorts_human_sra.csv     (all blood-mentioning human SRA + flags)
          tissue_cohorts_human_sra.csv     (disease-tissue human SRA + organ tag)
          viable_routes.csv                (diseases with blood AND tissue legs)

RECORD: if it errors, fix and re-run. Pure text-mining, no downloads.

Usage:  python 01b_mine_metadata.py [path/to/recount3_selection_*.csv]
"""
import glob, os, re, sys
import pandas as pd


# ---------------------------------------------------------------- locate input
def find_export():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    for pat in ("recount3_selection_*.csv", os.path.join("*", "recount3_selection_*.csv")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("recount3 export CSV not found. Pass it as an argument, or place "
             "recount3_selection_*.csv in the project root and re-run.")


CSV  = find_export()
ROOT = os.path.dirname(os.path.abspath(CSV)) or "."
OUT  = os.path.join(ROOT, "survey")
os.makedirs(OUT, exist_ok=True)
print(f"export : {CSV}")
print(f"output : {OUT}")

df = pd.read_csv(CSV)
h  = df[df.organism == "human"].copy()

# ---------------------------------------------------------- GTEx / TCGA catalog
gtex = sorted(h[h.project_home == "data_sources/gtex"].project.tolist())
tcga = sorted(h[h.project_home == "data_sources/tcga"].project.tolist())
pd.DataFrame({"gtex_tissue": gtex}).to_csv(os.path.join(OUT, "catalog_gtex_tissues.csv"), index=False)
pd.DataFrame({"tcga_project": tcga}).to_csv(os.path.join(OUT, "catalog_tcga_projects.csv"), index=False)
print(f"\nGTEx tissues ({len(gtex)}): {', '.join(gtex)}")
print(f"TCGA projects ({len(tcga)}): {', '.join(tcga)}")

# ----------------------------------------------------------------- SRA text pool
sra = h[h.project_home == "data_sources/sra"].copy()
sra["txt"] = (sra.study_title.fillna("") + " || " + sra.study_abstract.fillna("")).str.lower()

sra["whole_blood"]      = sra.txt.str.contains(r"whole blood")
sra["peripheral_blood"] = sra.txt.str.contains(r"peripheral blood(?! mononuclear)")
sra["pbmc"]             = sra.txt.str.contains(r"pbmc|peripheral blood mononuclear")
sra["platelet"]         = sra.txt.str.contains(r"platelet")
sra["is_tissue"]        = sra.txt.str.contains(r"biops|mucosa|resection|post-mortem|autopsy|\btissue\b|cortex|nucleus")
sra["is_cancer"]        = sra.txt.str.contains(r"cancer|carcinoma|tumou?r|adenocarcinoma|malignan|metasta|neoplas")

# ------------------------------------------------------------- disease dictionary
# disease -> (regex, GTEx tissue target, TCGA project(s) if cancer)
DISEASES = {
    "lung_cancer":          (r"non-small cell|\bnsclc\b|lung (adeno|squamous|cancer|carcinoma)", "LUNG", "LUAD/LUSC"),
    "colorectal_cancer":    (r"colorectal|colon (cancer|adeno|carcinoma)",                       "COLON", "COAD/READ"),
    "breast_cancer":        (r"breast (cancer|carcinoma|tumou?r)",                                "BREAST", "BRCA"),
    "liver_cancer_hcc":     (r"hepatocellular|\bhcc\b|liver cancer",                              "LIVER", "LIHC"),
    "prostate_cancer":      (r"prostate (cancer|carcinoma|adeno)",                                "PROSTATE", "PRAD"),
    "crohn_ibd":            (r"crohn|inflammatory bowel|\bibd\b",                                 "COLON/SMALL_INTESTINE", None),
    "ulcerative_colitis":   (r"ulcerative colitis",                                              "COLON", None),
    "nash_nafld":           (r"\bnash\b|\bnafld\b|steatohepat|non-?alcoholic fatty",              "LIVER", None),
    "ipf":                  (r"idiopathic pulmonary fibrosis|\bipf\b",                            "LUNG", None),
    "copd":                 (r"\bcopd\b|emphysema|chronic obstructive",                          "LUNG", None),
    "huntington":           (r"huntington",                                                      "BRAIN", None),
    "parkinson":            (r"parkinson",                                                        "BRAIN", None),
    "alzheimer":            (r"alzheimer",                                                        "BRAIN", None),
    "als":                  (r"amyotrophic|\bals\b",                                              "BRAIN/MUSCLE", None),
    "multiple_sclerosis":   (r"multiple sclerosis",                                              "BRAIN", None),
    "type1_diabetes":       (r"type 1 diabetes|type i diabetes|new-onset .*diabetes",            "PANCREAS", None),
    "sle_lupus":            (r"\blupus\b|\bsle\b|systemic lupus",                                 "KIDNEY", None),
    "rheumatoid_arthritis": (r"rheumatoid",                                                       None, None),
}


def tag_disease(t):
    return [d for d, (rx, _, _) in DISEASES.items() if re.search(rx, t)]


sra["diseases"] = sra.txt.map(tag_disease)

# ----------------------------------------------------------------- blood cohorts
blood = sra[sra.whole_blood | sra.peripheral_blood | sra.pbmc].copy()
blood["blood_type"] = [
    "whole_blood" if w else ("peripheral_blood" if p else "pbmc")
    for w, p in zip(blood.whole_blood, blood.peripheral_blood)
]
blood_out = blood.sort_values("n_samples", ascending=False)[
    ["project", "n_samples", "blood_type", "platelet", "diseases", "study_title"]
]
blood_out["diseases"] = blood_out.diseases.map(lambda x: ",".join(x))
blood_out.to_csv(os.path.join(OUT, "blood_cohorts_human_sra.csv"), index=False)

# ---------------------------------------------------------------- tissue cohorts
tissue = sra[sra.is_tissue & sra.diseases.map(bool)].copy()
tissue_out = tissue.sort_values("n_samples", ascending=False)[
    ["project", "n_samples", "is_cancer", "diseases", "study_title"]
]
tissue_out["diseases"] = tissue_out.diseases.map(lambda x: ",".join(x))
tissue_out.to_csv(os.path.join(OUT, "tissue_cohorts_human_sra.csv"), index=False)

# ------------------------------------------------------------- intersect -> routes
rows = []
for d, (rx, gtx, tc) in DISEASES.items():
    wb  = blood[(blood.blood_type == "whole_blood") & blood.diseases.map(lambda L: d in L)]
    ob  = blood[(blood.blood_type != "whole_blood") & blood.diseases.map(lambda L: d in L)]
    tis = tissue[tissue.diseases.map(lambda L: d in L)]
    has_tissue = (len(tis) > 0) or (tc is not None)
    if (len(wb) + len(ob)) == 0 or not has_tissue:
        continue
    top_wb = wb.sort_values("n_samples", ascending=False)
    rows.append({
        "disease": d,
        "gtex_tissue": gtx,
        "wholeblood_cohorts": len(wb),
        "wholeblood_top": (f"{top_wb.iloc[0].project} (n={int(top_wb.iloc[0].n_samples)})" if len(wb) else ""),
        "other_blood_cohorts": len(ob),   # peripheral/pbmc/platelet — off-match vs GTEx whole blood
        "tissue_source": (f"TCGA:{tc}" if tc else f"SRA:{len(tis)} studies"),
        "tissue_top": (f"{tis.sort_values('n_samples', ascending=False).iloc[0].project} "
                       f"(n={int(tis.sort_values('n_samples', ascending=False).iloc[0].n_samples)})" if len(tis) else ""),
        "blood_matches_gtex": len(wb) > 0,
    })

routes = pd.DataFrame(rows).sort_values(["blood_matches_gtex", "wholeblood_cohorts"], ascending=False)
routes.to_csv(os.path.join(OUT, "viable_routes.csv"), index=False)

print("\n" + "=" * 78)
print("VIABLE ROUTES  (disease with a blood leg AND a tissue leg in-snapshot)")
print("  blood_matches_gtex = a genuine WHOLE-BLOOD cohort exists (matches GTEx blood)")
print("  other_blood only   = blood cohort is platelet/PBMC/peripheral -> off-distribution")
print("=" * 78)
with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 40):
    print(routes.to_string(index=False))
print(f"\nwrote: viable_routes.csv, blood_cohorts_human_sra.csv, "
      f"tissue_cohorts_human_sra.csv, catalog_gtex_tissues.csv, catalog_tcga_projects.csv")
