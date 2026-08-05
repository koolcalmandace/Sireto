import os
import sys
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Project path resolution
_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1

out_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
if not out_path.exists():
    print("Error: Random sample results file not found on Desktop.")
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

total_records = len(df)
auto_df = df[df["v40_decision"] == "AUTO"]
review_df = df[df["v40_decision"] == "REVIEW"]
nomatch_df = df[df["v40_decision"] == "NO_MATCH"]

auto_count = len(auto_df)
review_count = len(review_df)
nomatch_count = len(nomatch_df)

auto_rate = auto_count / total_records
review_rate = review_count / total_records
nomatch_rate = nomatch_count / total_records

# Top-1 accuracy (any decision where we predicted the correct GT siret)
correct_matches = df[df["v40_siret_clean"] == df["gt_siret_clean"]]
overall_top1_accuracy = len(correct_matches) / total_records

# AUTO precision
auto_correct = auto_df[auto_df["v40_siret_clean"] == auto_df["gt_siret_clean"]]
auto_incorrect = auto_df[auto_df["v40_siret_clean"] != auto_df["gt_siret_clean"]]
auto_precision = len(auto_correct) / auto_count if auto_count > 0 else 0.0
fp_rate = len(auto_incorrect) / auto_count if auto_count > 0 else 0.0

# REVIEW/NO_MATCH details
review_correct = review_df[review_df["v40_siret_clean"] == review_df["gt_siret_clean"]]
review_incorrect = review_df[review_df["v40_siret_clean"] != review_df["gt_siret_clean"]]

print("--- GLOBAL STATISTICS ---")
print(f"Total Queries: {total_records}")
print(f"AUTO Rate: {auto_rate*100:.2f}% ({auto_count})")
print(f"REVIEW Rate: {review_rate*100:.2f}% ({review_count})")
print(f"NO_MATCH Rate: {nomatch_rate*100:.2f}% ({nomatch_count})")
print(f"Overall Top-1 Accuracy: {overall_top1_accuracy*100:.2f}% ({len(correct_matches)})")
print(f"AUTO Precision: {auto_precision*100:.2f}% ({len(auto_correct)}/{auto_count})")
print(f"False Positive Rate (Incorrect AUTOs): {fp_rate*100:.2f}% ({len(auto_incorrect)})")

print("\nRunning diagnostic checks on failures...")

# Initialize candidate store
partitions_dir = _PROJECT_ROOT / "data" / "candidates_v7_all"
store = PartitionedCandidateStore(partitions_dir)
config = RetrievalConfigV1()

failures = df[df["v40_siret_clean"] != df["gt_siret_clean"]].copy()
partition_cache = {}
print(f"Total failure cases: {len(failures)}")

db_missing_count = 0
early_elimination_count = 0
ranking_loss_count = 0
unexplained_count = 0

failure_details = []

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
        # No ground truth to match, not a true matching failure
        continue

    # Diagnostic check:
    # 1. Check if partition directories exist to avoid slow PyArrow tree scans on missing keys
    insee_dir = partitions_dir / "insee" / f"insee={insee}" if insee else None
    cp_dir = partitions_dir / "cp" / f"postcode={postcode}" if postcode else None
    
    insee_exists = insee_dir and insee_dir.exists()
    cp_exists = cp_dir and cp_dir.exists()
    
    if not insee_exists and not cp_exists:
        failure_details.append({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "gt_siret": gt_siret,
            "predicted_siret": predicted_siret,
            "decision": decision,
            "failure_type": "DATABASE_MISSING"
        })
        db_missing_count += 1
        continue

    # Load from store (guaranteed fast now because the directory exists)
    insee_cands = store.load_by_insee(insee) if insee_exists else []
    cp_cands = store.load_by_postcode(postcode) if cp_exists else []
    all_partition_cands = insee_cands + cp_cands
    partition_sirets = {clean_siret(c.get("siret")) for c in all_partition_cands if c.get("siret")}
    
    # 2. Run retrieval to see if it makes it to the candidate pool
    # Build crm_pre and run candidate pool
    crm_row = {
        "crm_name": crm_name,
        "crm_adresse": crm_address,
        "crm_cp": postcode,
        "crm_insee": insee,
    }
    
    from src.xgb_matcher.features import preprocess_crm_row
    crm_pre = preprocess_crm_row(crm_row)
    
    pool_res = build_candidate_pool(
        store=store,
        crm_row=crm_row,
        crm_pre=crm_pre,
        config=config,
        tfidf_cache={},
        partition_cache=partition_cache,
    )
    pool_cands = pool_res.candidates
    pool_sirets = {clean_siret(c.get("siret")) for c in pool_cands if c.get("siret")}
    
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

print(f"\nFailure Diagnostics Summary:")
print(f"  1. Database Missing Loss: {db_missing_count} (Company is physically not in the candidate partitions)")
print(f"  2. Early Elimination Loss: {early_elimination_count} (Company is in DB, but filtered out before Ranker)")
print(f"  3. Ranker Model Loss: {ranking_loss_count} (Company in pool, but ranked below someone else)")

# Write detailed reports to local artifact
report_path = Path(r"C:\Users\Kabouassi\.gemini\antigravity\brain\81356554-801e-4988-a5e6-00d0a75369b3\v4.0_random_sample_analysis_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# Version 4.0 Random Sample Evaluation & Diagnostic Report\n\n")
    f.write("## Executive Summary\n")
    f.write(f"- **Total Queries Analyzed**: {total_records}\n")
    f.write(f"- **Automation (AUTO) Rate**: {auto_rate*100:.2f}% ({auto_count} queries)\n")
    f.write(f"- **Manual Review Rate**: {review_rate*100:.2f}% ({review_count} queries)\n")
    f.write(f"- **No Match Rate**: {nomatch_rate*100:.2f}% ({nomatch_count} queries)\n")
    f.write(f"- **Overall Top-1 Match Accuracy**: {overall_top1_accuracy*100:.2f}% ({len(correct_matches)} queries)\n")
    f.write(f"- **AUTO Precision (Accuracy of Auto Matches)**: {auto_precision*100:.2f}% ({len(auto_correct)}/{auto_count})\n")
    f.write(f"- **False Positive Rate**: {fp_rate*100:.2f}% ({len(auto_incorrect)} incorrect matches in AUTO)\n\n")
    
    f.write("## Failure Diagnostic Analysis\n")
    f.write(f"We analyzed the {len(failures)} failure cases where the predicted SIRET did not match the ground truth:\n\n")
    f.write(f"| Failure Category | Count | % of Failures | Description |\n")
    f.write(f"| :--- | :---: | :---: | :--- |\n")
    total_diagnosed = db_missing_count + early_elimination_count + ranking_loss_count
    pct_db = (db_missing_count / total_diagnosed * 100) if total_diagnosed > 0 else 0
    pct_early = (early_elimination_count / total_diagnosed * 100) if total_diagnosed > 0 else 0
    pct_rank = (ranking_loss_count / total_diagnosed * 100) if total_diagnosed > 0 else 0
    f.write(f"| **Database Missing Loss** | {db_missing_count} | {pct_db:.1f}% | Company is physically not in the candidate partitions (out of scope) |\n")
    f.write(f"| **Early Elimination Loss** | {early_elimination_count} | {pct_early:.1f}% | Company is in DB, but filtered out in Retrieval stage |\n")
    f.write(f"| **Ranker Model Loss** | {ranking_loss_count} | {pct_rank:.1f}% | Company was in candidate pool, but XGBoost ranked another candidate higher |\n\n")
    
    f.write("## Failure Log (Detailed Mismatch Samples)\n")
    f.write("| CRM ID | CRM Company Name | Ground Truth SIRET | Predicted SIRET | Decision | Diagnostic Classification |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for fd in failure_details[:50]: # Show up to top 50 in table
        f.write(f"| `{fd['crm_id']}` | {fd['crm_name']} | `{fd['gt_siret']}` | `{fd['predicted_siret']}` | `{fd['decision']}` | **{fd['failure_type']}** |\n")

print(f"\nSaved full diagnostic markdown report to: {report_path}")
