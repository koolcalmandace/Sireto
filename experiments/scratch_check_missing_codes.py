import pandas as pd
from pathlib import Path

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)
insee_col = "sirene_insee" if "sirene_insee" in df.columns else ("crm_insee" if "crm_insee" in df.columns else "insee")
postcode_col = "sirene_cp" if "sirene_cp" in df.columns else ("crm_cp" if "crm_cp" in df.columns else "postcode")

crm_insee_codes = set(df[insee_col].dropna().unique())
crm_postcodes = set(df[postcode_col].dropna().unique())

base_dir = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_all")
existing_insee = set()
if (base_dir / "insee").exists():
    # Parquet hive partitioning directory format: insee=XXXXX
    for p in (base_dir / "insee").iterdir():
        if p.is_dir() and p.name.startswith("insee="):
            existing_insee.add(p.name.split("=")[1])

existing_cp = set()
if (base_dir / "cp").exists():
    for p in (base_dir / "cp").iterdir():
        if p.is_dir() and p.name.startswith("postcode="):
            existing_cp.add(p.name.split("=")[1])

missing_insee = crm_insee_codes - existing_insee
missing_cp = crm_postcodes - existing_cp

print(f"CRM INSEE: {len(crm_insee_codes)}, Existing: {len(existing_insee)}, Missing: {len(missing_insee)}")
print(f"CRM Postcodes: {len(crm_postcodes)}, Existing: {len(existing_cp)}, Missing: {len(missing_cp)}")
print("Missing INSEE codes:", sorted(list(missing_insee)))
print("Missing Postcodes:", sorted(list(missing_cp)))
