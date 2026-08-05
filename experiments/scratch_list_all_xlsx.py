import os
from pathlib import Path

desktop = Path(r"C:\Users\Kabouassi\Desktop")
print("All XLSX/CSV on Desktop:")
for item in desktop.glob("*"):
    if item.suffix.lower() in [".xlsx", ".csv"]:
        print(f"  {item.name} (Size: {item.stat().st_size / 1024:.1f} KB)")
