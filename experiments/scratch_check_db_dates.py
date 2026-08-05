import os
import datetime
from pathlib import Path

p_dir = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data")

files = [
    "StockEtablissement_utf8.parquet",
    "StockUniteLegale_utf8.parquet",
    "harvest_full.sqlite"
]

for f in files:
    fp = p_dir / f
    if fp.exists():
        stat = fp.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / (1024 * 1024)
        print(f"File: {f}")
        print(f"  Modified Time: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"File: {f} -> NOT FOUND")
