import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

fp = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\src\xgb_matcher\infer.py"

if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "decider_path" in line or "class TwoStageModelProfile" in line:
            print(f"Line {idx+1}: {line}")
else:
    print("File not found")
