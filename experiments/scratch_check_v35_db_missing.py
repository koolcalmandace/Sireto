import pandas as pd
import sys
from pathlib import Path

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))

v35_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v35_random.xlsx")
df = pd.read_excel(v35_path)

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

def clean_siret(val):
    return clean_code(val, 14)

df["gt_siret_clean"] = df["gt_siret"].apply(clean_siret)

# We want to check how many gt_sirets in the v35 sample are missing from the partitioned database
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)

missing_count = 0
in_scope_count = 0

for idx, row in df.iterrows():
    postcode = clean_code(row.get("crm_cp"), 5)
    insee = clean_code(row.get("crm_insee"), 5)
    gt_siret = row.get("gt_siret_clean")
    
    if not gt_siret:
        continue
        
    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    cp_dir = partitions_dir / "cp" / f"postcode={postcode}" if postcode else None
    
    insee_exists = insee_dir and insee_dir.exists()
    cp_exists = cp_dir and cp_dir.exists()
    
    if not insee_exists and not cp_exists:
        missing_count += 1
    else:
        # Load partition and check
        insee_cands = store.load_by_insee(insee) if insee_exists else []
        cp_cands = store.load_by_postcode(postcode) if cp_exists else []
        all_partition_cands = insee_cands + cp_cands
        partition_sirets = {clean_siret(c.get("siret")) for c in all_partition_cands if c.get("siret")}
        
        if gt_siret not in partition_sirets:
            missing_count += 1
        else:
            in_scope_count += 1

print(f"V3.5 Random Sample:")
print(f"  Total records: {len(df)}")
print(f"  In-scope queries (GT in DB): {in_scope_count} ({in_scope_count/len(df)*100:.1f}%)")
print(f"  Out-of-scope queries (GT missing from DB): {missing_count} ({missing_count/len(df)*100:.1f}%)")
