import pandas as pd
print("GTEx Phenotypes head:")
df_pheno = pd.read_csv('data/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt', sep='\t', nrows=5)
print(df_pheno)
print("\nGTEX_data head:")
df_data = pd.read_csv('data/GTEX_data.csv', nrows=5)
print(df_data.iloc[:, :10]) # print first 10 columns only
print("Number of columns:", len(df_data.columns))
print("Columns contains 'Age'?:", 'Age' in df_data.columns)
print("Columns contains 'Sex'?:", 'Sex' in df_data.columns)
print("Columns contains 'tissue'?:", 'tissue' in df_data.columns or 'SMTSD' in df_data.columns or 'Tissue' in df_data.columns)
