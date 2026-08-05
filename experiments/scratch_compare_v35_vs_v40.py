import pandas as pd
import sys
from pathlib import Path

v35_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v35_random.xlsx")
v40_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")

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

def analyze_file(path, prefix):
    if not path.exists():
        return None
    df = pd.read_excel(path)
    df["gt_siret_clean"] = df["gt_siret"].apply(clean_siret)
    
    # Auto-detect prediction column (v35_siret or v40_siret or similar)
    pred_col = None
    dec_col = None
    score_col = None
    for c in df.columns:
        if c.endswith("_siret") and c.startswith(("v3", "v4")):
            pred_col = c
        if c.endswith("_decision") and c.startswith(("v3", "v4")):
            dec_col = c
        if c.endswith("_score") and c.startswith(("v3", "v4")):
            score_col = c
            
    if not pred_col:
        # Fallback to general names
        pred_col = "chosen_siret_final" if "chosen_siret_final" in df.columns else "siret"
    if not dec_col:
        dec_col = "xgb_status" if "xgb_status" in df.columns else "decision"
    if not score_col:
        score_col = "routing_confidence" if "routing_confidence" in df.columns else "score"
        
    df["pred_siret_clean"] = df[pred_col].apply(clean_siret)
    
    total = len(df)
    correct_matches = df[df["pred_siret_clean"] == df["gt_siret_clean"]]
    accuracy = len(correct_matches) / total
    
    auto_df = df[df[dec_col] == "AUTO"]
    auto_count = len(auto_df)
    auto_rate = auto_count / total
    
    auto_correct = auto_df[auto_df["pred_siret_clean"] == auto_df["gt_siret_clean"]]
    auto_incorrect = auto_df[auto_df["pred_siret_clean"] != auto_df["gt_siret_clean"]]
    auto_precision = len(auto_correct) / auto_count if auto_count > 0 else 0.0
    fp_rate = len(auto_incorrect) / auto_count if auto_count > 0 else 0.0
    
    same_siren = 0
    diff_siren = 0
    for idx, row in auto_incorrect.iterrows():
        gt = row["gt_siret_clean"]
        pred = row["pred_siret_clean"]
        if gt and pred and len(gt) == 14 and len(pred) == 14:
            if gt[:9] == pred[:9]:
                same_siren += 1
            else:
                diff_siren += 1
        else:
            diff_siren += 1
            
    return {
        "total": total,
        "accuracy": accuracy,
        "correct": len(correct_matches),
        "auto_rate": auto_rate,
        "auto_count": auto_count,
        "auto_precision": auto_precision,
        "fp_rate": fp_rate,
        "fp_count": len(auto_incorrect),
        "same_siren": same_siren,
        "diff_siren": diff_siren
    }

r35 = analyze_file(v35_path, "v35")
r40 = analyze_file(v40_path, "v40")

print("--- COMPARATIVE SUMMARY ---")
if r35:
    print("Version 3.5 (Legacy Weights):")
    print(f"  Total Queries: {r35['total']}")
    print(f"  Overall Top-1 Accuracy: {r35['accuracy']*100:.2f}% ({r35['correct']})")
    print(f"  AUTO Rate: {r35['auto_rate']*100:.2f}% ({r35['auto_count']})")
    print(f"  AUTO Precision: {r35['auto_precision']*100:.2f}%")
    print(f"  False Positive Rate: {r35['fp_rate']*100:.2f}% ({r35['fp_count']})")
    print(f"    - Same SIREN (Sister Branch / HQ): {r35['same_siren']} ({r35['same_siren']/r35['fp_count']*100:.1f}%)" if r35['fp_count'] > 0 else "")
    print(f"    - Different SIREN (Unrelated Firm): {r35['diff_siren']} ({r35['diff_siren']/r35['fp_count']*100:.1f}%)" if r35['fp_count'] > 0 else "")
else:
    print("Version 3.5 results file not found.")

if r40:
    print("\nVersion 4.0 (New SSOT Weights):")
    print(f"  Total Queries: {r40['total']}")
    print(f"  Overall Top-1 Accuracy: {r40['accuracy']*100:.2f}% ({r40['correct']})")
    print(f"  AUTO Rate: {r40['auto_rate']*100:.2f}% ({r40['auto_count']})")
    print(f"  AUTO Precision: {r40['auto_precision']*100:.2f}%")
    print(f"  False Positive Rate: {r40['fp_rate']*100:.2f}% ({r40['fp_count']})")
    print(f"    - Same SIREN (Sister Branch / HQ): {r40['same_siren']} ({r40['same_siren']/r40['fp_count']*100:.1f}%)" if r40['fp_count'] > 0 else "")
    print(f"    - Different SIREN (Unrelated Firm): {r40['diff_siren']} ({r40['diff_siren']/r40['fp_count']*100:.1f}%)" if r40['fp_count'] > 0 else "")
