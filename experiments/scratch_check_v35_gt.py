import pandas as pd
from pathlib import Path

v35_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v35_random.xlsx")
df = pd.read_excel(v35_path)

print("V3.5 Columns:", list(df.columns))
print("gt_siret head:")
print(df["gt_siret"].head(10).tolist())
print("gt_siret null count:", df["gt_siret"].isna().sum())

# Check how many rows have non-null gt_siret
non_null = df[df["gt_siret"].notna()]
print("Non-null gt_siret count:", len(non_null))
