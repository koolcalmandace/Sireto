import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1

partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)
config = RetrievalConfigV1()

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

crm_row = {
    "crm_name": "MISSION LOCALE INSERTION FORMATION EMPLOI",
    "crm_adresse": "12 RUE ELMERICH",
    "crm_cp": "80080",
    "crm_insee": "80021",
}
gt_siret = "32679473200031"

insee = clean_code(crm_row["crm_insee"], 5)
postcode = clean_code(crm_row["crm_cp"], 5)

print(f"Cleaned codes: insee={insee}, postcode={postcode}")

insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
cp_dir = partitions_dir / "cp" / f"postcode={postcode}" if postcode else None

insee_exists = insee_dir and insee_dir.exists()
cp_exists = cp_dir and cp_dir.exists()

print(f"Directory existence: insee_exists={insee_exists}, cp_exists={cp_exists}")

insee_cands = store.load_by_insee(insee) if insee_exists else []
cp_cands = store.load_by_postcode(postcode) if cp_exists else []
all_partition_cands = insee_cands + cp_cands
partition_sirets = {clean_siret(c.get("siret")) for c in all_partition_cands if c.get("siret")}

print(f"Total candidates in partition: {len(all_partition_cands)}")
print(f"Is gt_siret {gt_siret} in partition_sirets? {gt_siret in partition_sirets}")

# Let's run build_candidate_pool
from src.xgb_matcher.features import preprocess_crm_row
crm_pre = preprocess_crm_row(crm_row)

pool_res = build_candidate_pool(
    store=store,
    crm_row=crm_row,
    crm_pre=crm_pre,
    config=config,
    tfidf_cache={},
    partition_cache={},
)
pool_cands = pool_res.candidates
pool_sirets = {clean_siret(c.get("siret")) for c in pool_cands if c.get("siret")}

print(f"Total candidates in pool: {len(pool_cands)}")
print(f"Is gt_siret {gt_siret} in pool_sirets? {gt_siret in pool_sirets}")
print(f"Loss reason from pool build: {pool_res.loss_reason}")
print(f"gt_in_base_pool: {pool_res.gt_in_base_pool}")
print(f"gt_in_filtered_pool: {pool_res.gt_in_filtered_pool}")
print(f"gt_in_tfidf_pool: {pool_res.gt_in_tfidf_pool}")
