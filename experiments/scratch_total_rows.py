import pandas as pd
import sqlite3

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)
insee_col = "sirene_insee" if "sirene_insee" in df.columns else ("crm_insee" if "crm_insee" in df.columns else "insee")
insee_codes = sorted(list(df[insee_col].dropna().unique()))

batch = insee_codes[:100]

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

placeholders = ",".join(["?"] * len(batch))
cursor.execute(f"SELECT count(*) FROM etablissements WHERE commune IN ({placeholders});", batch)
total_count = cursor.fetchone()[0]

print(f"Total establishments in first 100 communes: {total_count}")
conn.close()
