import pandas as pd
import sys
from pathlib import Path

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))

out_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
df = pd.read_excel(out_path)

def clean_siret(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(14)
    return s

df["gt_siret_clean"] = df["gt_siret"].apply(clean_siret)
df["v40_siret_clean"] = df["v40_siret"].apply(clean_siret)

auto_df = df[df["v40_decision"] == "AUTO"]
auto_incorrect = auto_df[auto_df["v40_siret_clean"] != auto_df["gt_siret_clean"]].copy()

print(f"Total Incorrect AUTOs: {len(auto_incorrect)}")
print("Scores of Incorrect AUTOs:")
print("Min score:", auto_incorrect["v40_score"].min())
print("Max score:", auto_incorrect["v40_score"].max())
print("Mean score:", auto_incorrect["v40_score"].mean())

print("\nDetail of top 10 incorrect AUTOs:")
cols_to_print = ["crm_name", "crm_adresse", "gt_siret_clean", "v40_match_name", "v40_siret_clean", "v40_score"]
print(auto_incorrect[cols_to_print].head(10).to_dict(orient="records"))
