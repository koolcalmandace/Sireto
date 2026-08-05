import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

fp = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\train_xgb_decider.py"

if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "meta[" in line or "meta_dict[" in line or "models_dir" in line or "save_model" in line or ".json" in line:
            print(f"Line {idx+1}: {line}")
else:
    print("File not found")
