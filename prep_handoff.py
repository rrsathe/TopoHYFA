import pandas as pd
import numpy as np
import os

def main():
    # Target Genes
    target_genes = [
        "CHDH", "SREBF1", "CA14", "CYP2J2", "CTF1", "SNX22", "ETNPPL", "SYNPO2L", 
        "ARHGAP1", "FAM220A", "HBA2", "BLM", "MAFK", "HMGN2", "C4orf46"
    ]
    
    out_dir = 'Imputation/output/HYFA_export'
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading GTEx phenotypes...")
    pheno_file = 'data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt'
    df_pheno = pd.read_csv(pheno_file, sep='\t')
    
    print("Loading GTEx expression data...")
    expr_file = 'data/GTEX_data.csv'
    df_expr = pd.read_csv(expr_file, index_col=0)
    
    print(f"Expression data shape: {df_expr.shape}")
    
    # Filter for 'Heart - Left Ventricle' or similar
    all_tissues = df_expr['tissue'].unique()
    target_tissue = None
    for t in all_tissues:
        t_str = str(t).lower()
        if t == 'Heart_L_Vent':
            target_tissue = t
            break
            
    if target_tissue is None:
        print(f"Error: Could not find 'Heart - Left Ventricle' in tissues. Available tissues: {all_tissues}")
        return
        
    print(f"Found tissue match: '{target_tissue}'")
    mask_heart = df_expr['tissue'] == target_tissue
    df_heart = df_expr[mask_heart].copy()
    
    # Extract subjects
    # GTEx sample ID in index, e.g., GTEX-XXXXX-YYYY-SM-ZZZZ 
    df_heart['SUBJID'] = [ "-".join(idx.split("-")[:2]) for idx in df_heart.index ]
    
    # Extract features for target genes
    col_map = {str(c).upper(): c for c in df_heart.columns}
    
    for g in target_genes:
        if g.upper() in col_map:
            col_name = col_map[g.upper()]
            # ensure it's named exactly as in target_genes for the export
            df_heart[g] = df_heart[col_name]
        else:
            print(f"Warning: Gene {g} not found. Filling with zeros.")
            df_heart[g] = 0.0
            
    df_heart_targets = df_heart[target_genes].astype(float)
    
    # Compute 15x15 Pearson correlation
    corr_matrix = df_heart_targets.corr(method='pearson')
    # Fill NaNs with 0 (which happens for columns with all zeros)
    corr_matrix = corr_matrix.fillna(0)
    # Set negative correlations to 0
    corr_matrix[corr_matrix < 0] = 0
    
    # Save target expressions and adj matrix
    df_heart_targets.to_csv(os.path.join(out_dir, 'target_genes_15.csv'), index=False)
    corr_matrix.to_csv(os.path.join(out_dir, 'adjacency_matrix.csv'), index=True)
    
    # Construct confounders
    df_conf = pd.merge(df_heart[['SUBJID']], df_pheno, on='SUBJID', how='inner')
    
    # Keep Age and Sex if present
    cols_to_keep = ['SUBJID']
    if 'AGE' in df_conf.columns:
        cols_to_keep.append('AGE')
    if 'SEX' in df_conf.columns:
        cols_to_keep.append('SEX')
        
    df_conf = df_conf[cols_to_keep]
    df_conf.to_csv(os.path.join(out_dir, 'confounders.csv'), index=False)
    
    print(f"Successfully exported data for {len(df_heart_targets)} samples and 15 target genes.")

if __name__ == "__main__":
    main()
