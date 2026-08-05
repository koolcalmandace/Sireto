import pandas as pd
import sys
from pathlib import Path

base_path = Path(r"C:\Users\Kabouassi\Desktop\Clean House\Project North Star\Results\sample")
desktop = Path(r"C:\Users\Kabouassi\Desktop")

files = [
    (base_path / "results_1000_sample_v21_cleaned.xlsx", "V2.1"),
    (base_path / "results_1000_sample_v24.xlsx", "V2.4"),
    (base_path / "results_1000_sample_v26_random.xlsx", "V2.6"),
    (base_path / "results_1000_sample_v30_random.xlsx", "V3.0"),
    (base_path / "results_1000_sample_v31_restored_random.xlsx", "V3.1"),
    (base_path / "results_1000_sample_v33_random.xlsx", "V3.3"),
    (base_path / "results_1000_sample_v34_random.xlsx", "V3.4"),
    (desktop / "results_1000_sample_v35_random.xlsx", "V3.5"),
    (desktop / "results_1000_sample_v40_random.xlsx", "V4.0")
]

def clean_siret(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(14)
    return s

results = []

for fp, name in files:
    if not fp.exists():
        print(f"Skipping {name}: not found at {fp}")
        continue
    try:
        df = pd.read_excel(fp)
        df["gt_siret_clean"] = df["gt_siret"].apply(clean_siret)
        
        # Detect prediction column
        pred_col = None
        dec_col = None
        for c in df.columns:
            if c.endswith("_siret") and c.startswith(("v2", "v3", "v4")):
                pred_col = c
            if c.endswith("_decision") and c.startswith(("v2", "v3", "v4")):
                dec_col = c
        
        # Fallbacks for older files
        if not pred_col:
            for c in ["chosen_siret_final", "chosen_siret", "siret"]:
                if c in df.columns:
                    pred_col = c
                    break
        if not dec_col:
            for c in ["xgb_status", "xgb_decision", "decision", "status"]:
                if c in df.columns:
                    dec_col = c
                    break
                    
        if not pred_col or not dec_col:
            print(f"Skipping {name}: columns not found. Available: {list(df.columns)}")
            continue
            
        df["pred_siret_clean"] = df[pred_col].apply(clean_siret)
        
        total = len(df)
        correct = len(df[df["pred_siret_clean"] == df["gt_siret_clean"]])
        accuracy = correct / total
        
        auto_df = df[df[dec_col] == "AUTO"]
        auto_count = len(auto_df)
        auto_rate = auto_count / total
        
        auto_correct = len(auto_df[auto_df["pred_siret_clean"] == auto_df["gt_siret_clean"]])
        auto_precision = auto_correct / auto_count if auto_count > 0 else 0.0
        fp_count = auto_count - auto_correct
        fp_rate = fp_count / auto_count if auto_count > 0 else 0.0
        
        # Sister branch check
        same_siren = 0
        diff_siren = 0
        for idx, row in auto_df[auto_df["pred_siret_clean"] != auto_df["gt_siret_clean"]].iterrows():
            gt = row["gt_siret_clean"]
            pred = row["pred_siret_clean"]
            if gt and pred and len(gt) == 14 and len(pred) == 14:
                if gt[:9] == pred[:9]:
                    same_siren += 1
                else:
                    diff_siren += 1
            else:
                diff_siren += 1
                
        results.append({
            "version": name,
            "total": total,
            "accuracy": f"{accuracy*100:.1f}% ({correct})",
            "auto_rate": f"{auto_rate*100:.1f}% ({auto_count})",
            "auto_precision": f"{auto_precision*100:.1f}%",
            "fp_count": fp_count,
            "same_siren": same_siren,
            "diff_siren": diff_siren
        })
    except Exception as e:
        print(f"Error processing {name}: {e}")

summary_df = pd.DataFrame(results)
print("\n--- HISTORICAL BENCHMARK COMPARISON ---")
print(summary_df.to_string(index=False))
summary_df.to_csv("historical_benchmark_comparison.csv", index=False)
