import json
import pandas as pd
from pathlib import Path

report_path = Path(r"C:\Users\Kabouassi\.gemini\antigravity\brain\81356554-801e-4988-a5e6-00d0a75369b3\ranking_overtake_deep_analysis.json")
if not report_path.exists():
    print("Report JSON file not found!")
    exit(1)

with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total overtakes analyzed: {len(data)}")

df = pd.DataFrame(data)
if not df.empty:
    print("\n=== CLASSIFICATION SUMMARY ===")
    print(df["status"].value_counts().to_string())
    
    print("\n=== TOP FEATURES DRIVING THE OVERTAKE ===")
    feature_counts = {}
    for r in data:
        for feat, diff in r["advantages"]:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    sorted_feats = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    for feat, count in sorted_feats[:10]:
        print(f"  {feat}: {count} times ({count/len(data)*100:.1f}%)")
