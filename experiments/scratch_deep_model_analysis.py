import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import json

# Project path resolution
_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import preprocess_crm_row, make_features_from_preprocessed
from src.xgb_matcher.profile import InferenceProfile

# 1. Load latest ranker model
model_dir = _PROJECT_ROOT / "models"
metas = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
if not metas:
    print("Error: No metadata file found in models/")
    sys.exit(1)

with open(metas[0], "r", encoding="utf-8") as f:
    meta = json.load(f)

# Find latest ranker path relative to project root
ranker_name = meta["ranker_model"]
ranker_path = model_dir / ranker_name
ranker = xgb.Booster()
ranker.load_model(str(ranker_path))

# Features expected by ranker
ranker_features = meta["ranker_feature_order"]

# 2. Initialize partitions
partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)
config = RetrievalConfigV1()

# 3. Read random sample results
out_path = Path(r"C:\Users\Kabouassi\Desktop\Clean House\Project North Star\Results\sample\results_1000_sample_v40_random.xlsx")
if not out_path.exists():
    print(f"Error: Results file not found at {out_path}")
    sys.exit(1)
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

analysis_records = []

tfidf_cache = {}
partition_cache = {}

print("=== STARTING DEEP ANALYSIS OF RANKING OVERTAKES ===")
sys.stdout.flush()

for idx, row in failures.iterrows():
    gt_siret = row["gt_siret_clean"]
    predicted_siret = row["v40_siret_clean"]
    
    if not gt_siret or not predicted_siret:
        continue
        
    crm_name = row.get("crm_name")
    crm_address = row.get("crm_adresse")
    postcode = clean_code(row.get("crm_cp"), 5)
    insee = clean_code(row.get("crm_insee"), 5)
    
    # Check if both are in partition
    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    if not insee_dir or not insee_dir.exists():
        continue
        
    try:
        df_part = pd.read_parquet(insee_dir)
    except Exception:
        continue
        
    # Get candidates
    gt_rows = df_part[df_part["siret"] == gt_siret]
    pred_rows = df_part[df_part["siret"] == predicted_siret]
    
    if gt_rows.empty or pred_rows.empty:
        continue
        
    gt_cand = gt_rows.iloc[0].to_dict()
    pred_cand = pred_rows.iloc[0].to_dict()
    
    # Sanitize float NaNs to None for naming.py compatibility
    gt_cand = {k: (None if pd.isna(v) else v) for k, v in gt_cand.items()}
    pred_cand = {k: (None if pd.isna(v) else v) for k, v in pred_cand.items()}
    
    # Skip if GT is closed (intentional loss, not a ranker error)
    if gt_cand.get("etat_admin") != "A":
        continue
        
    # Compute features
    crm_row = {
        "crm_name": crm_name,
        "crm_adresse": crm_address,
        "crm_cp": postcode,
        "crm_insee": insee,
    }
    crm_pre = preprocess_crm_row(crm_row)
    
    gt_feats = make_features_from_preprocessed(crm_pre, gt_cand)
    pred_feats = make_features_from_preprocessed(crm_pre, pred_cand)
    
    # Format for XGBoost DMatrix
    df_gt = pd.DataFrame([gt_feats])[ranker_features].fillna(0.0)
    df_pred = pd.DataFrame([pred_feats])[ranker_features].fillna(0.0)
    
    d_gt = xgb.DMatrix(df_gt, feature_names=ranker_features)
    d_pred = xgb.DMatrix(df_pred, feature_names=ranker_features)
    
    gt_score = float(ranker.predict(d_gt)[0])
    pred_score = float(ranker.predict(d_pred)[0])
    
    # Verify indeed predicted has higher score
    if pred_score <= gt_score:
        continue
        
    # Compute differences (Pred - GT)
    diffs = {}
    for f in ranker_features:
        diffs[f] = float(pred_feats.get(f, 0.0) - gt_feats.get(f, 0.0))
        
    # Sort features by where predicted had the biggest advantage
    sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
    top_advantages = [x for x in sorted_diffs if x[1] > 0.01][:3]
    
    # Categorize the overtake justification
    is_sister_branch = gt_cand.get("siren") == pred_cand.get("siren")
    
    is_postcode_mismatch = (
        crm_row["crm_cp"] is not None and
        pred_cand.get("postcode") == crm_row["crm_cp"] and
        gt_cand.get("postcode") != crm_row["crm_cp"]
    )
    
    warranted_status = "UNWARRANTED (Model skew / Feature imbalance)"
    warranted_reason = "Model favored weaker matches due to feature weights."
    
    if is_postcode_mismatch:
        warranted_status = "JUSTIFIED BY DIRTY INPUT (CRM Postcode matches Predicted, mismatches GT)"
        warranted_reason = f"CRM input has postcode {crm_row['crm_cp']} which physically belongs to the predicted candidate ({pred_cand.get('postcode')}) not the correct GT candidate ({gt_cand.get('postcode')})."
    elif is_sister_branch:
        warranted_status = "AMBIGUOUS SISTER BRANCH (Same SIREN, different SIRET)"
        warranted_reason = "The correct entity has the exact same name/brand/siren but is a different establishment. The model failed to resolve the local address difference."
    elif diffs.get("name_jaro", 0.0) > 0.15:
        warranted_status = "JUSTIFIED BY LEXICAL SIMILARITY (Predicted name is closer to CRM input)"
        warranted_reason = f"CRM input name '{crm_name}' matches predicted name '{pred_cand.get('denomination_ul') or pred_cand.get('denomination')}' much better than GT name '{gt_cand.get('denomination_ul') or gt_cand.get('denomination')}'."

    record = {
        "crm_name": crm_name,
        "gt_name": gt_cand.get('denomination_ul') or gt_cand.get('denomination'),
        "pred_name": pred_cand.get('denomination_ul') or pred_cand.get('denomination'),
        "gt_siret": gt_siret,
        "pred_siret": predicted_siret,
        "gt_score": gt_score,
        "pred_score": pred_score,
        "score_diff": pred_score - gt_score,
        "status": warranted_status,
        "reason": warranted_reason,
        "advantages": top_advantages,
        "gt_features": {f: float(gt_feats.get(f, 0.0)) for f in ranker_features},
        "pred_features": {f: float(pred_feats.get(f, 0.0)) for f in ranker_features}
    }
    
    analysis_records.append(record)

print(f"Analyzed {len(analysis_records)} ranker overtake cases.")

# Write full details to JSON report
report_path = Path(r"C:\Users\Kabouassi\.gemini\antigravity\brain\81356554-801e-4988-a5e6-00d0a75369b3\ranking_overtake_deep_analysis.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(analysis_records, f, indent=2)

print(f"Detailed JSON analysis written to {report_path}")

# Print high level summary
df_sum = pd.DataFrame(analysis_records)
if not df_sum.empty:
    print("\n=== SUMMARY OF RANKER OVERTAKE ROOT CAUSES ===")
    print(df_sum["status"].value_counts().to_string())
    print("\nSample detailed breakdown:")
    for idx, r in df_sum.head(3).iterrows():
        print(f"\nQuery: {r['crm_name']}")
        print(f"  GT:   {r['gt_name']} ({r['gt_siret']}) | Score: {r['gt_score']:.4f}")
        print(f"  Pred: {r['pred_name']} ({r['pred_siret']}) | Score: {r['pred_score']:.4f}")
        print(f"  Classification: {r['status']}")
        print(f"  Explanation:    {r['reason']}")
        print(f"  Top advantages of Predicted over GT:")
        for feat, adv in r["advantages"]:
            print(f"    - {feat}: +{adv:.4f}")
sys.stdout.flush()
