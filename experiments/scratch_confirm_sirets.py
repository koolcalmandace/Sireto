import sqlite3
import pandas as pd
import re
from pathlib import Path
import unicodedata

# Paths
db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"

# Normalization function
def clean_name(name):
    if not name or pd.isna(name):
        return ""
    # Lowercase
    s = str(name).lower()
    # Normalize accents
    s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # Remove punctuation and non-alphanumeric chars
    s = re.sub(r"[^\w\s]", "", s)
    # Remove common French legal forms and suffixes
    suffixes = [
        r"\bsas\b", r"\bsarl\b", r"\bsa\b", r"\bsci\b", r"\beurl\b", r"\bste\b", 
        r"\bsociete\b", r"\bcoop\b", r"\bcooperative\b", r"\bassociation\b", 
        r"\bass\b", r"\bets\b", r"\betablissement\b", r"\bgroup\b", r"\bgroupe\b",
        r"\bfrance\b"
    ]
    for suf in suffixes:
        s = re.sub(suf, "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", "", s)
    return s.strip()

print("Loading CRM database...")
df = pd.read_csv(crm_path, sep=";", dtype=str)
print(f"Loaded {len(df)} rows from CRM.")

print("Connecting to harvest_full.sqlite...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

exact_clean_matches = []
exact_raw_matches = []
loose_matches_90 = []

# Fetching all GT SIRETs from SQLite
sirets = df["gt_siret"].dropna().unique().tolist()
print(f"Querying {len(sirets)} unique ground truth SIRETs in SQLite...")

db_name_map = {}
# Chunk queries to prevent SQLite variable limits
chunk_size = 900
for i in range(0, len(sirets), chunk_size):
    chunk = sirets[i:i+chunk_size]
    placeholders = ",".join(["?"] * len(chunk))
    query = f"""
    SELECT 
        e.siret, e.nom_commercial, e.liste_enseignes,
        ent.nom_raison_sociale, ent.nom_complet, ent.sigle
    FROM etablissements e
    LEFT JOIN entreprises ent ON e.siren = ent.siren
    WHERE e.siret IN ({placeholders})
    """
    cursor.execute(query, chunk)
    for r in cursor.fetchall():
        siret, nom_comm, enseignes, raison_sociale, nom_complet, sigle = r
        # Combine all possible names in DB
        names = []
        if nom_comm: names.append(nom_comm)
        if raison_sociale: names.append(raison_sociale)
        if nom_complet: names.append(nom_complet)
        if sigle: names.append(sigle)
        if enseignes:
            try:
                import json
                ens_list = json.loads(enseignes)
                if isinstance(ens_list, list):
                    names.extend(ens_list)
                else:
                    names.append(str(enseignes))
            except Exception:
                names.append(str(enseignes))
        
        db_name_map[siret] = [str(n) for n in names if n]

print("Performing name comparisons...")
exact_count = 0
clean_count = 0

confirmed_rows = []

for idx, row in df.iterrows():
    crm_name = row.get("crm_name")
    gt_siret = row.get("gt_siret")
    
    if not gt_siret or gt_siret not in db_name_map:
        continue
        
    db_names = db_name_map[gt_siret]
    
    # 1. Raw exact match
    is_exact_raw = False
    for db_n in db_names:
        if str(crm_name).strip().upper() == str(db_n).strip().upper():
            is_exact_raw = True
            break
            
    # 2. Clean exact match (normalized names match exactly)
    is_clean_exact = False
    matched_db_name = None
    cleaned_crm = clean_name(crm_name)
    for db_n in db_names:
        if cleaned_crm == clean_name(db_n):
            is_clean_exact = True
            matched_db_name = db_n
            break
            
    if is_exact_raw:
        exact_count += 1
        exact_raw_matches.append(row.to_dict())
    if is_clean_exact:
        clean_count += 1
        row_dict = row.to_dict()
        row_dict["sirene_name"] = matched_db_name
        exact_clean_matches.append(row_dict)
        confirmed_rows.append(row_dict)

print("\n=== CONFIRMATION RESULTS ===")
print(f"1. Pure Raw Exact Match (identical strings): {exact_count} rows ({exact_count/len(df)*100:.1f}%)")
print(f"2. Cleaned Exact Match (normalized strings):  {clean_count} rows ({clean_count/len(df)*100:.1f}%)")

# Save the cleaned exact match dataset as our new expanded confirmed dataset
out_csv = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt_confirmed_expanded.csv")
df_confirmed = pd.DataFrame(confirmed_rows)
df_confirmed.to_csv(out_csv, sep=";", index=False)
print(f"\nSaved {len(df_confirmed)} confirmed clean rows to {out_csv}")

conn.close()
