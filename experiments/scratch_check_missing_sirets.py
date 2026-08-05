import sqlite3
import pandas as pd
from pathlib import Path
import json

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Test sample SIRETs from the failure report
test_cases = [
    {"name": "MISSION LOCALE INSERTION FORMATION EMPLOI", "siret": "32679473200031"},
    {"name": "PLANETE SCIENCES MIDI PYRENEES ASSOCIA", "siret": "39507200200045"},
    {"name": "CHANTRERIE POLYTECH", "siret": "19440984300753"},
    {"name": "IDEF 86", "siret": "26860086300016"},
    {"name": "Woonoz", "siret": "48452879900022"},
]

print("=== SQLite Lookup ===")
for tc in test_cases:
    siret = tc["siret"]
    cursor.execute("SELECT siret, siren, commune, code_postal, etat_administratif, nom_commercial FROM etablissements WHERE siret = ?;", (siret,))
    row = cursor.fetchone()
    if row:
        print(f"SIRET {siret} FOUND in SQLite:")
        print(f"  Siren: {row[1]}, Commune: {row[2]}, CP: {row[3]}, Etat Admin: {row[4]}, Name: {row[5]}")
        
        # Check if the partition for this commune exists in candidates_v7_all
        commune = row[2]
        part_path = Path(f"C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/candidates_v7_all/insee/insee={commune}")
        if part_path.exists():
            print(f"  Partition directory insee={commune} EXISTS.")
            # Read parquet files to see if SIRET is in them
            try:
                df_part = pd.read_parquet(part_path)
                siret_matches = df_part[df_part["siret"] == siret]
                if not siret_matches.empty:
                    print(f"    -> SIRET is present in the partition file!")
                else:
                    print(f"    -> SIRET is MISSING from the partition file! (Partition has {len(df_part)} rows)")
            except Exception as e:
                print(f"    -> Error reading partition: {e}")
        else:
            print(f"  Partition directory insee={commune} does NOT exist.")
    else:
        print(f"SIRET {siret} NOT found in SQLite.")
    print("-" * 50)

conn.close()
