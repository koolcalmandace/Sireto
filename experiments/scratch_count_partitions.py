import os
from pathlib import Path

# Count how many unique commune partitions exist in candidates_v7_all
store_path = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_all")

# Partitions are stored as insee=XXXXX subdirectories
insee_dirs = [d for d in store_path.iterdir() if d.is_dir() and d.name.startswith("insee=")]
print(f"Partition store (candidates_v7_all):")
print(f"  Unique INSEE communes covered: {len(insee_dirs):,}")

# Quick sample
sample = sorted([d.name.replace("insee=", "") for d in insee_dirs[:5]])
print(f"  Sample codes: {sample}")

# France total communes reference
print(f"\nFrance national context:")
print(f"  Total communes in France (approx): ~35,000")
print(f"  Partition store covers: {len(insee_dirs):,} communes")
print(f"  Coverage ratio: {len(insee_dirs)/35000*100:.1f}% of all French communes")
