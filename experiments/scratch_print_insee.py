import pandas as pd

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)
insee_col = "sirene_insee" if "sirene_insee" in df.columns else ("crm_insee" if "crm_insee" in df.columns else "insee")
insee_codes = sorted(list(df[insee_col].dropna().unique()))

print("Total unique INSEE codes:", len(insee_codes))
print("First 20 INSEE codes:", insee_codes[:20])
