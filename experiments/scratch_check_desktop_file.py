import os
from pathlib import Path

path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
print(f"File exists: {path.exists()}")
if path.exists():
    print(f"File size: {os.path.getsize(path)} bytes")
