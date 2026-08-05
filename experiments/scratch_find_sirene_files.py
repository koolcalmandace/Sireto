import os
from pathlib import Path

# Search parent folders or home folders for the Stock files
paths_to_search = [
    Path(r"C:\Users\Kabouassi"),
    Path(r"C:\Users\Kabouassi\.gemini\antigravity"),
    Path(r"C:\Users\Kabouassi\Desktop"),
]

print("Searching for SIRENE raw files...")

found = False
for search_dir in paths_to_search:
    if not search_dir.exists():
        continue
    print(f"Scanning {search_dir}...")
    for item in search_dir.rglob("StockEtablissement_utf8.parquet"):
        print("FOUND:", item)
        print("  Size:", item.stat().st_size / (1024 * 1024), "MB")
        print("  Modified:", datetime.datetime.fromtimestamp(item.stat().st_mtime) if 'datetime' in globals() else "")
        found = True

if not found:
    print("No raw SIRENE parquet files found in home or desktop.")
