import sqlite3
import pandas as pd
from pathlib import Path

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

out_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v40_random.xlsx")
df = pd.read_excel(out_path)

def clean_code(val, length):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        s = s.zfill(length)
    return s

df["gt_siret_clean"] = df["gt_siret"].apply(lambda x: clean_code(x, 14))
df["v40_siret_clean"] = df["v40_siret"].apply(lambda x: clean_code(x, 14))

failures = df[df["v40_siret_clean"] != df["gt_siret_clean"]].copy()

results = []

sqlite_missing = 0
insee_missing_partition = 0
siret_missing_in_partition = 0
siret_found_in_partition = 0

for idx, row in failures.iterrows():
    gt_siret = row["gt_siret_clean"]
    if not gt_siret:
        continue
        
    # 1. Check if SIRET exists in SQLite
    cursor.execute("SELECT commune, code_postal, etat_administratif FROM etablissements WHERE siret = ?;", (gt_siret,))
    db_row = cursor.fetchone()
    
    if not db_row:
        sqlite_missing += 1
        results.append((gt_siret, "MISSING_FROM_SQLITE"))
        continue
        
    commune = db_row[0]
    cp = db_row[1]
    etat_admin = db_row[2]
    
    # 2. Check if the partition directory exists
    part_path = Path(f"C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/candidates_v7_all/insee/insee={commune}")
    if not part_path.exists():
        insee_missing_partition += 1
        results.append((gt_siret, "PARTITION_DIR_MISSING"))
        continue
        
    # 3. Check if the SIRET is in the partition parquet file
    try:
        df_part = pd.read_parquet(part_path, columns=["siret"])
        if gt_siret in df_part["siret"].values:
            siret_found_in_partition += 1
            results.append((gt_siret, f"FOUND_IN_PARTITION_ETAT_{etat_admin}"))
        else:
            siret_missing_in_partition += 1
            results.append((gt_siret, f"MISSING_FROM_PARTITION_FILE_ETAT_{etat_admin}"))
    except Exception as e:
        results.append((gt_siret, f"ERROR_READING_PARTITION_{str(e)}"))

print("=== CORRECTED RAW DATA DIAGNOSTICS ===")
print(f"Total Matching Failures Checked: {len(results)}")
print(f"1. Physically missing from SQLite:      {sqlite_missing}")
print(f"2. Partition directory missing:          {insee_missing_partition}")
print(f"3. In SQLite, but missing from Partition: {siret_missing_in_partition}")
print(f"4. Found in Partition file:              {siret_found_in_partition}")

# Print detail on why they were missing from partition file
print("\nDetail of results:")
df_res = pd.DataFrame(results, columns=["siret", "status"])
print(df_res["status"].value_counts())

conn.close()
