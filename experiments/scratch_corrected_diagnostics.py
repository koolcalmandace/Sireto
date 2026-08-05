import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

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

db_missing_count = 0
early_elimination_count = 0
ranking_loss_count = 0
unexplained_count = 0

failure_details = []

tfidf_cache = {}
partition_cache = {}

for idx, row in failures.iterrows():
    crm_id = row.get("crm_id")
    crm_name = row.get("crm_name")
    crm_address = row.get("crm_adresse")
    postcode = clean_code(row.get("crm_cp"), 5)
    insee = clean_code(row.get("crm_insee"), 5)
    gt_siret = row.get("gt_siret_clean")
    predicted_siret = row.get("v40_siret_clean")
    decision = row.get("v40_decision")
    
    if not gt_siret:
        continue

    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    cp_dir = partitions_dir / "cp" / f"postcode={postcode}" if postcode else None
    
    insee_exists = insee_dir and insee_dir.exists()
    cp_exists = cp_dir and cp_dir.exists()
    
    # Check if target SIRET is physically in the database
    # Let's check partition store
    insee_cands = store.load_by_insee(insee) if insee_exists else []
    cp_cands = store.load_by_postcode(postcode) if cp_exists else []
    all_partition_cands = insee_cands + cp_cands
    partition_sirets = {clean_siret(c.get("siret")) for c in all_partition_cands if c.get("siret")}
    
    crm_row = {
        "crm_name": crm_name,
        "crm_adresse": crm_address,
        "crm_cp": postcode,
        "crm_insee": insee,
    }
    crm_pre = preprocess_crm_row(crm_row)
    
    # IMPORTANT: Pass gt_norm=gt_siret to get the true base and filtered pool flags!
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
    
    failure_type = "UNEXPLAINED"
    if gt_siret not in partition_sirets:
        failure_type = "DATABASE_MISSING"
        db_missing_count += 1
    elif gt_siret not in pool_sirets:
        failure_type = "EARLY_ELIMINATION"
        early_elimination_count += 1
    else:
        failure_type = "RANKER_LOSS"
        ranking_loss_count += 1
        
    failure_details.append({
        "crm_id": crm_id,
        "crm_name": crm_name,
        "gt_siret": gt_siret,
        "predicted_siret": predicted_siret,
        "decision": decision,
        "failure_type": failure_type
    })

print("=== CORRECTED FAILURE DIAGNOSTICS ===")
print(f"Total True Matching Failures: {len(failure_details)}")
print(f"Database Missing Loss: {db_missing_count} ({db_missing_count/len(failure_details)*100:.1f}%)")
print(f"Early Elimination Loss: {early_elimination_count} ({early_elimination_count/len(failure_details)*100:.1f}%)")
print(f"Ranker Model Loss: {ranking_loss_count} ({ranking_loss_count/len(failure_details)*100:.1f}%)")

# Sample Ranker Loss rows
print("\nSample Ranker Loss rows:")
ranker_failures = [fd for fd in failure_details if fd["failure_type"] == "RANKER_LOSS"]
for rf in ranker_failures[:10]:
    print(f"  Name: {rf['crm_name']} | GT: {rf['gt_siret']} | Pred: {rf['predicted_siret']} | Decision: {rf['decision']}")
