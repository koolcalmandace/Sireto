import pandas as pd
from pathlib import Path

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)

insee_col = "sirene_insee" if "sirene_insee" in df.columns else "crm_insee"
postcode_col = "sirene_cp" if "sirene_cp" in df.columns else "crm_cp"

crm_insee_codes = set(df[insee_col].dropna().unique())
crm_postcodes = set(df[postcode_col].dropna().unique())

print(f"Total unique INSEE codes in CRM: {len(crm_insee_codes)}")
print(f"Total unique postcodes in CRM: {len(crm_postcodes)}")

# Let's count how many partitions exist in candidates_v7_all/insee
insee_dir = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_all\insee")
existing_insee = {d.name.split("=")[1] for d in insee_dir.iterdir() if d.is_dir() and d.name.startswith("insee=")}

cp_dir = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_all\cp")
existing_cp = {d.name.split("=")[1] for d in cp_dir.iterdir() if d.is_dir() and d.name.startswith("postcode=")}

print(f"Existing INSEE partitions in candidates_v7_all: {len(existing_insee)}")
print(f"Existing postcode partitions in candidates_v7_all: {len(existing_cp)}")

missing_insee = crm_insee_codes - existing_insee
missing_cp = crm_postcodes - existing_cp

print(f"Missing INSEE codes in candidates_v7_all: {len(missing_insee)}")
print(f"Missing postcodes in candidates_v7_all: {len(missing_cp)}")
print("Sample missing INSEE:", sorted(list(missing_insee))[:20])
print("Sample missing postcodes:", sorted(list(missing_cp))[:20])
