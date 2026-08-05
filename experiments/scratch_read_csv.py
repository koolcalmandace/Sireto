import pandas as pd
import time
import sys

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"

# Test sep=";"
try:
    t0 = time.time()
    df = pd.read_csv(crm_path, sep=";", dtype=str)
    print(f"Read with sep=';' succeeded in {time.time() - t0:.2f} seconds. Shape: {df.shape}")
    print("Columns:", list(df.columns[:5]))
except Exception as e:
    print(f"sep=';' failed: {e}")

# Test sep=","
try:
    t0 = time.time()
    df = pd.read_csv(crm_path, sep=",", dtype=str)
    print(f"Read with sep=',' succeeded in {time.time() - t0:.2f} seconds. Shape: {df.shape}")
    print("Columns:", list(df.columns[:5]))
except Exception as e:
    print(f"sep=',' failed: {e}")
