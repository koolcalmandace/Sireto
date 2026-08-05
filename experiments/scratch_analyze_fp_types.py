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

same_siren_count = 0
diff_siren_count = 0

for idx, row in auto_incorrect.iterrows():
    gt = row["gt_siret_clean"]
    pred = row["v40_siret_clean"]
    if gt and pred and len(gt) == 14 and len(pred) == 14:
        gt_siren = gt[:9]
        pred_siren = pred[:9]
        if gt_siren == pred_siren:
            same_siren_count += 1
        else:
            diff_siren_count += 1
    else:
        diff_siren_count += 1

print(f"Total Incorrect AUTOs: {len(auto_incorrect)}")
print(f"  - Same SIREN (Sister Branch / HQ): {same_siren_count} ({same_siren_count/len(auto_incorrect)*100:.1f}%)")
print(f"  - Different SIREN (Unrelated Firm): {diff_siren_count} ({diff_siren_count/len(auto_incorrect)*100:.1f}%)")
