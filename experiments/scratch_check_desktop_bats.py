import os
from pathlib import Path

desktop = Path(r"C:\Users\Kabouassi\Desktop")
for f in desktop.glob("*.bat"):
    print(f"File: {f.name}")
    try:
        with open(f, encoding="utf-8-sig", errors="ignore") as file:
            print(file.read())
            print("="*40)
    except Exception as e:
        print(f"Error reading {f.name}: {e}")
