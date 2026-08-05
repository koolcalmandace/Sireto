import pandas as pd
from pathlib import Path

out_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
df = pd.read_excel(out_path)

test_sirets = ["32679473200031", "39507200200045", "19440984300753", "26860086300016", "48452879900022"]

# Let's clean the GT siret column to match
def clean_code(val, length):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(length)
    return s

df["gt_siret_clean"] = df["gt_siret"].apply(lambda x: clean_code(x, 14))

print("=== Failure rows in Excel ===")
for s in test_sirets:
    row = df[df["gt_siret_clean"] == s]
    if not row.empty:
        r = row.iloc[0]
        print(f"Name: {r.get('crm_name')}")
        print(f"  crm_cp in Excel:    {r.get('crm_cp')}")
        print(f"  crm_insee in Excel: {r.get('crm_insee')}")
        print(f"  gt_siret:           {r.get('gt_siret')}")
        print(f"  v40_siret:          {r.get('v40_siret')}")
        print(f"  v40_decision:       {r.get('v40_decision')}")
    else:
        print(f"SIRET {s} not found in Excel.")
    print("-" * 50)
