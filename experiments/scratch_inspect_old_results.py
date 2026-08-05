import os
from pathlib import Path

p = Path(r"C:\Users\Kabouassi\Desktop\Clean House\Project North Star\Results")
if not p.exists():
    print(f"Directory not found: {p}")
    exit(1)

print(f"Scanning directory: {p}")
for root, dirs, files in os.walk(p):
    for f in files:
        if f.endswith((".xlsx", ".csv")) and not f.startswith("~$"):
            fp = Path(root) / f
            rel = fp.relative_to(p)
            print(f"File: {rel} (Size: {fp.stat().st_size / 1024:.1f} KB)")
