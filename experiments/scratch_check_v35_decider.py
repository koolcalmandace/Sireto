import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

fp = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\infer_xgb_two_stage_v35.py"

if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "decider" in line.lower() or "booster" in line.lower():
            print(f"Line {idx+1}: {line}")
else:
    print("File not found")
