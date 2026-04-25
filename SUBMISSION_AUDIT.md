# Submission Audit: Topology-Aware HYFA Project

## A. Issues Found

### Critical

- The original `python train_gtex.py --interpretability` command is incomplete. The script requires `--target-genes`, `--topology-matrix`, and `--confounders`.
- The README examples use `"Whole Blood"`, but the local GTEx data uses `"Whole_Blood"`. Using the wrong tissue string can silently produce empty or ineffective validation/evaluation paths.
- The project claims a 15-gene target set, but the local file contains 14 target columns and only 11 are found in the processed GTEx matrix. Missing from local processed data: `CA14`, `ETNPPL`, `SYNPO2L`.
- The completed full training run used `--lambda-reg 0.0`, so it validates HYFA without topology regularization. Claims about topology-aware HYFA improving biomarkers are MISSING / UNCERTAIN until a positive-`lambda_reg` run is evaluated.
- Current side-by-side results do not support "HYFA has better average performance than TEEBoT" on this 11-gene run. TEEBoT mean Pearson is higher: `0.3694` versus HYFA `0.3369`.
- The downstream SEX AUC result should not be described as a strong improvement. HYFA imputed AUC is `0.4458`, blood surrogate AUC is `0.4465`, and ground-truth heart AUC is `0.4253`.

### Major

- The TEEBoT baseline implemented here is a simplified PCA + linear regression baseline. It is valid as a student baseline, but it is not a full reproduction of TEEBoT because it does not include splicing, LLR filtering, or the original full covariate design.
- The test sample size is small: 55 test participants with only 11 female cases and 44 male controls for SEX prediction. AUC estimates are noisy and should be presented as exploratory.
- No confidence intervals, bootstrap intervals, or repeated train/test seeds are reported. Claims should be descriptive, not statistically conclusive.
- The graph smoothness regularizer is conceptually reasonable, but the adjacency source must be described plainly as a co-expression/topology matrix from the selected genes. It should not be called a validated causal biological network.

### Minor

- The repository contains legacy notebooks and patch scripts that distract from the student submission.
- Interpretability hooks are useful, but they explain the linear decoder/metagene contribution structure, not causal gene importance.
- The README states "zero padding" and "15 genes"; in the executed pipeline the usable target set is 11 genes after matching the processed GTEx matrix.

## B. Fixed Methodology

Use a simple, defensible pipeline:

1. Load processed GTEx v8 expression and phenotype metadata.
2. Split by donor using existing files: `data/splits/gtex_train.txt`, `data/splits/gtex_val.txt`, `data/splits/gtex_test.txt`.
3. Select target genes from `Imputation/output/HYFA_export/target_genes_15.csv`.
4. Keep only genes present in the processed GTEx matrix. Report this as 11 usable genes.
5. Train HYFA from `Whole_Blood` to `Heart_L_Vent`.
6. Optionally train TopoHYFA by setting `--lambda-reg > 0`.
7. Evaluate per-gene Pearson correlation and RMSE on held-out test donors.
8. Compare with a transparent PCA + linear regression baseline.
9. Run downstream SEX prediction only as an exploratory sanity check.
10. Save interpretability tensors and simple top-gene contribution plots.

## C. Minimal Working Code

Use the new runner:

```bash
python student_pipeline.py
```

For topology-aware HYFA:

```bash
python student_pipeline.py --lambda-reg 0.1
```

To reuse an existing trained `data/model.pth`:

```bash
python student_pipeline.py --skip-train
```

The runner calls the existing scripts in order:

- `train_gtex.py`
- `eval_15.py`
- `benchmark_teebot.py`
- `run_disease_prediction.py`
- `visualize_interpretability.py`

It also sets:

- `WANDB_MODE=disabled`
- `MPLCONFIGDIR=/tmp/matplotlib`
- `--num-workers 0`

This avoids the multiprocessing socket failure observed in the local sandbox/WSL environment.

## D. Final Results Interpretation

Verified results from the completed run:

- Training completed for 200 epochs.
- Model checkpoint saved to `data/model.pth`.
- Evaluation target: `Whole_Blood -> Heart_L_Vent`.
- Usable genes: 11.
- HYFA mean Pearson: `0.3369`.
- TEEBoT-style PCA + linear baseline mean Pearson: `0.3694`.
- HYFA improved over TEEBoT on 5 of 11 genes: `SREBF1`, `CYP2J2`, `BLM`, `HMGN2`, `C4orf46`.
- CYP2J2 improved in this HYFA run relative to the PCA baseline: `0.4109` vs `0.2564`.
- TEEBoT-style baseline was better on 6 of 11 genes and had better average Pearson/RMSE.
- SEX prediction is not strong evidence of clinical utility: HYFA imputed AUC `0.4458`, blood surrogate AUC `0.4465`, ground truth heart AUC `0.4253`.

Interpretability output:

- `gene_metagene_attribution`: `(55, 11, 4)`
- `prediction_uncertainty`: `(55, 11)`
- 55 PNG contribution plots were generated.
- For batch 0, sample 0, top contribution gene was `HMGN2`.

## E. Submission-Ready Summary

This project studies whether heart left ventricle gene expression can be inferred from whole blood transcriptome data in GTEx v8. The biological motivation is reasonable: heart tissue is difficult to collect, while blood is accessible, and prior work such as TEEBoT showed that whole blood expression can predict tissue-specific expression for many genes. HYFA is also a reasonable comparison because it was designed for multi-tissue gene expression imputation using donor, tissue, and metagene representations.

The final student version should present the work as a targeted imputation experiment, not as a full clinical biomarker system. The data are split by donor into train, validation, and test sets, which is the correct split level for avoiding donor leakage. The task is to train on whole blood expression and predict heart left ventricle expression for a small cardiovascular gene panel. Although the intended panel was described as 15 genes, the local processed data supports 11 usable genes, and the report should state this clearly.

The baseline is a simple TEEBoT-style model: PCA on whole blood expression followed by linear regression to predict target tissue expression. This is a valid and explainable student baseline, but it is not a complete reproduction of the published TEEBoT method because splicing and likelihood-ratio filtering are not included. HYFA is trained as the neural model. The topology-aware extension adds graph smoothness regularization using an adjacency matrix over the selected genes. This is a reasonable idea because it encourages related genes to have smoother learned decoder weights, but it should be described as a regularized model, not as proof of causal biological network structure.

The current verified results are mixed. HYFA improves specific genes such as CYP2J2, but the PCA baseline has better average Pearson correlation across the 11 genes. Therefore, the honest conclusion is that HYFA may help selected biomarkers but does not dominate the simpler baseline on this small target panel. The downstream SEX prediction experiment should be treated as exploratory only, because all AUCs are below 0.5 and the sample size is small.

The strongest defensible claim is:

> We implemented a reproducible targeted gene-expression imputation pipeline comparing a transparent PCA + linear baseline with HYFA, and added an optional topology-aware regularizer and interpretability outputs. On the current 11-gene heart panel, HYFA improved selected genes including CYP2J2, while the simple baseline remained stronger on average. This suggests topology-aware HYFA is a promising but not yet conclusively superior extension.

Recommended final scope for viva:

- Keep: data split, 11-gene target panel, PCA baseline, HYFA, optional topology regularization, Pearson/RMSE evaluation, simple interpretability plots.
- Remove or downplay: strong clinical claims, "imputed better than ground truth", disease prediction as a main result, claims of global superiority.
- State MISSING / UNCERTAIN: topology-aware positive-lambda comparison unless a separate `--lambda-reg 0.1` run is completed and reported.

## Sources Checked

- HYFA paper: https://www.nature.com/articles/s42256-023-00684-8
- TEEBoT paper DOI: https://doi.org/10.1126/sciadv.abd6991
- TEEBoT bibliographic/abstract page: https://experts.illinois.edu/en/publications/predicting-tissue-specific-gene-expression-from-whole-blood-trans/
