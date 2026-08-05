import os
from pathlib import Path

desktop = Path(r"C:\Users\Kabouassi\Desktop")
print("Listing files on Desktop matching results_*.xlsx:")
for item in desktop.glob("results_*.xlsx"):
    print(f"  {item.name} (Size: {item.stat().st_size / 1024:.1f} KB)")

project_data = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data")
print("\nListing files in project data directory matching *.xlsx or *.csv:")
for item in project_data.glob("results_*"):
    print(f"  {item.name}")
