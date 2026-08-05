import pandas as pd
from pathlib import Path

# Check how many CRM rows have dept 92 or dept 06 codes that are missing
crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)

insee_col = "sirene_insee" if "sirene_insee" in df.columns else "crm_insee"
postcode_col = "sirene_cp" if "sirene_cp" in df.columns else "crm_cp"

# Missing codes we found
missing_insee = {'06030', '92002', '92007', '92009', '92014', '92019', '92020', '92023', '92024',
                 '92032', '92035', '92036', '92040', '92044', '92046', '92049', '92051', '92064',
                 '92071', '92072', '92073', '92075'}
missing_cp = {'06100', '06110', '06130', '06150', '0[ND]', '92110', '92120', '92130', '92140',
              '92150', '92160', '92170', '92190', '92200', '92210', '92220', '92230', '92240',
              '92250', '92260', '92270', '92290', '92300', '92310', '92320', '92330', '92340'}

rows_with_missing_insee = df[df[insee_col].isin(missing_insee)]
rows_with_missing_cp = df[df[postcode_col].isin(missing_cp)]

# Union by index
affected = df.index.isin(rows_with_missing_insee.index) | df.index.isin(rows_with_missing_cp.index)
affected_df = df[affected]

print(f"CRM rows affected by missing codes: {len(affected_df)} out of {len(df)} ({100*len(affected_df)/len(df):.1f}%)")
print("\nBreakdown by department:")
dept_counts = affected_df[insee_col].str[:2].value_counts()
print(dept_counts)

print("\nSample affected rows:")
print(affected_df[["crm_name", insee_col, postcode_col]].head(10).to_string())
