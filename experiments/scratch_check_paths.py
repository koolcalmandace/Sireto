import json
from pathlib import Path

meta_v40 = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\models\xgb_two_stage_meta_20260730_155642.json")
if meta_v40.exists():
    with open(meta_v40) as f:
        print("V4.0 Meta:", json.load(f))
else:
    print("V4.0 Meta not found")

# Look at V3.5 script to see where it loads candidates from
v35_script = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\infer_xgb_two_stage_v35.py")
if v35_script.exists():
    with open(v35_script, encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "partitions" in line or "data" in line:
            print(f"V3.5 Line {idx+1}: {line}")
