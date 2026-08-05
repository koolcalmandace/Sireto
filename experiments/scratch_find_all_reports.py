import os
from pathlib import Path

p = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data")
for sub in ["reports", "reports_normalized"]:
    d = p / sub
    if d.exists():
        print(f"Contents of {sub}:")
        for item in d.glob("*"):
            print(f"  {item.name} (Size: {item.stat().st_size / 1024:.1f} KB)")
