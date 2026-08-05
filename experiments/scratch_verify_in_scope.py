import pandas as pd
import sys
from pathlib import Path

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore

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

failures = df[df["v40_siret_clean"] != df["gt_siret_clean"]].copy()

# Initialize store
partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)

in_scope_failures = []

for idx, row in failures.iterrows():
    postcode = clean_siret(row.get("crm_cp"))
    if postcode and len(postcode) > 5:
        postcode = postcode[:5]
    insee = clean_siret(row.get("crm_insee"))
    gt_siret = row.get("gt_siret_clean")
    
    if not gt_siret:
        continue
        
    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    cp_dir = partitions_dir / "cp" / f"postcode={postcode}" if postcode else None
    
    insee_exists = insee_dir and insee_dir.exists()
    cp_exists = cp_dir and cp_dir.exists()
    
    if insee_exists or cp_exists:
        # Load candidates
        insee_cands = store.load_by_insee(insee) if insee_exists else []
        cp_cands = store.load_by_postcode(postcode) if cp_exists else []
        all_partition_cands = insee_cands + cp_cands
        partition_sirets = {clean_siret(c.get("siret")) for c in all_partition_cands if c.get("siret")}
        
        if gt_siret in partition_sirets:
            in_scope_failures.append({
                "crm_id": row.get("crm_id"),
                "crm_name": row.get("crm_name"),
                "gt_siret": gt_siret,
                "v40_siret": row.get("v40_siret_clean"),
                "decision": row.get("v40_decision")
            })

print(f"Total failures: {len(failures)}")
print(f"In-scope failures (where GT SIRET is in partition but match failed): {len(in_scope_failures)}")
if in_scope_failures:
    print("Samples of in-scope failures:")
    for f in in_scope_failures[:5]:
        print(f)
