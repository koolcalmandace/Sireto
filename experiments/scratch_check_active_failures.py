import sys
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import preprocess_crm_row, make_features_from_preprocessed
import joblib

partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)
config = RetrievalConfigV1()

# Load decider and ranker models
model_dir = _PROJECT_ROOT / "models"
decider = joblib.load(model_dir / "decider_v40.joblib")
ranker = joblib.load(model_dir / "ranker_v40.joblib")

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

results = []

# Let's inspect 10 active failures to see exactly what happened
active_checked = 0

tfidf_cache = {}
partition_cache = {}

for idx, row in failures.iterrows():
    gt_siret = row["gt_siret_clean"]
    if not gt_siret:
        continue
        
    crm_name = row.get("crm_name")
    crm_address = row.get("crm_adresse")
    postcode = clean_code(row.get("crm_cp"), 5)
    insee = clean_code(row.get("crm_insee"), 5)
    predicted_siret = row.get("v40_siret_clean")
    
    # Check if this company is active in partition
    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    if not insee_dir or not insee_dir.exists():
        continue
        
    df_part = pd.read_parquet(insee_dir)
    gt_rows = df_part[df_part["siret"] == gt_siret]
    if gt_rows.empty:
        continue
        
    gt_cand = gt_rows.iloc[0].to_dict()
    if gt_cand.get("etat_admin") != "A":
        continue
        
    # We found an active failure!
    active_checked += 1
    if active_checked > 10:
        break
        
    print(f"\n--- ACTIVE FAILURE #{active_checked}: {crm_name} ---")
    print(f"  CRM Address: {crm_address} | CP: {postcode} | INSEE: {insee}")
    print(f"  GT SIRET: {gt_siret} | Name in DB: {gt_cand.get('denomination_ul') or gt_cand.get('denomination')}")
    print(f"  Predicted SIRET: {predicted_siret}")
    
    # Let's rebuild the pool
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
    
    print(f"  GT in pool? {gt_siret in pool_sirets} (Pool size: {len(pool_res.candidates)})")
    
    if gt_siret not in pool_sirets:
        print(f"  -> ELIMINATED early in retrieval! Reason: {pool_res.loss_reason}")
    else:
        # GT is in pool, check model scoring!
        print("  -> GT made it to pool! Checking XGBoost scoring...")
        
        # Build features for all candidates in the pool
        features_list = []
        cands_in_pool = pool_res.candidates
        for c in cands_in_pool:
            feats = make_features_from_preprocessed(crm_pre, c)
            # Add identifiers for debugging
            feats["_siret"] = c.get("siret")
            feats["_name"] = c.get("denomination_ul") or c.get("denomination")
            features_list.append(feats)
            
        df_feats = pd.DataFrame(features_list)
        
        # Score using ranker
        feature_cols = [col for col in df_feats.columns if not col.startswith("_")]
        # Fill NaNs
        df_feats_fill = df_feats[feature_cols].fillna(0.0)
        df_feats["ranker_score"] = ranker.predict_proba(df_feats_fill)[:, 1]
        
        # Sort by score
        df_feats = df_feats.sort_values(by="ranker_score", ascending=False).reset_index(drop=True)
        
        print("  Top 5 candidates ranked by XGBoost:")
        for rank, r in df_feats.head(5).iterrows():
            is_gt = " (GROUND TRUTH)" if r["_siret"] == gt_siret else ""
            is_pred = " (PREDICTED)" if r["_siret"] == predicted_siret else ""
            print(f"    #{rank+1}: {r['_name']} ({r['_siret']}) | Score: {r['ranker_score']:.4f}{is_gt}{is_pred}")
            
print("\nActive failures diagnostics complete!")
