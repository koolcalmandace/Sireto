import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

fp = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\generate_training_samples_v5fast.py"

if os.path.exists(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        if "def generate_split" in line:
            print(f"Line {idx+1}: {line}")
            for j in range(idx + 60, min(len(lines), idx + 120)):
                print(f"  {j+1}: {lines[j]}")
else:
    print("File not found")
