import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Temporarily increase cache size to 1000
import src.xgb_matcher.partitioned_store
src.xgb_matcher.partitioned_store._MAX_STORE_CACHE_SIZE = 1000

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import preprocess_crm_row

partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)
config = RetrievalConfigV1()

out_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
df = pd.read_excel(out_path)

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
df["v40_siret_clean"] = df["v40_siret"].apply(clean_siret)

failures = df[df["v40_siret_clean"] != df["gt_siret_clean"]].copy()

early_elimination = 0
ranker_loss = 0
failures_checked = 0

tfidf_cache = {}
partition_cache = {}

print(f"Starting analysis of {len(failures)} failures...")

for idx, row in failures.iterrows():
    gt_siret = row["gt_siret_clean"]
    predicted_siret = row["v40_siret_clean"]
    if not gt_siret:
        continue
        
    crm_name = row.get("crm_name")
    crm_address = row.get("crm_adresse")
    postcode = clean_code(row.get("crm_cp"), 5)
    insee = clean_code(row.get("crm_insee"), 5)
    
    # 1. Use the store lazy-loader directly (which is now cached with size 1000)
    insee_cands = store.load_by_insee(insee) if insee else []
    if not insee_cands:
        continue
        
    # Check if active and present
    gt_cand = None
    for c in insee_cands:
        if clean_siret(c.get("siret")) == gt_siret:
            gt_cand = c
            break
            
    if not gt_cand or gt_cand.get("etat_admin") != "A":
        continue
        
    failures_checked += 1
    
    # Rebuild candidate pool (will reuse the cache we just populated!)
    crm_row = {
        "crm_name": crm_name,
        "crm_adresse": crm_address,
        "crm_cp": postcode,
        "crm_insee": insee,
    }
    crm_pre = preprocess_crm_row(crm_row)
    
    pool_res = build_candidate_pool(
        store=store,
        crm_row=crm_row,
        crm_pre=crm_pre,
        config=config,
        tfidf_cache=tfidf_cache,
        partition_cache=partition_cache,
        gt_siret=gt_siret
    )
    
    pool_sirets = {clean_siret(c.get("siret")) for c in pool_res.candidates if c.get("siret")}
    
    if gt_siret not in pool_sirets:
        early_elimination += 1
    else:
        ranker_loss += 1

print("\n=== ACTIVE FAILURES DIAGNOSTICS ===")
print(f"Total Active failures checked: {failures_checked}")
print(f"1. Early Elimination Loss: {early_elimination} ({early_elimination/failures_checked*100:.1f}%)")
print(f"2. Ranker Model Loss:      {ranker_loss} ({ranker_loss/failures_checked*100:.1f}%)")
