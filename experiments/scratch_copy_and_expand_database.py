import os
import shutil
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Paths
base_dir = Path("data/candidates_v7_all")
expanded_dir = Path("data/candidates_v7_expanded")
crm_path = "data/crm_ok_gt.csv"
harvest_db_path = "data/harvest_full.sqlite"

print("1. Performing clean copy of baseline candidates_v7_all to candidates_v7_expanded...")
if expanded_dir.exists():
    print("Deleting existing partial candidates_v7_expanded folder...")
    shutil.rmtree(expanded_dir)

shutil.copytree(base_dir, expanded_dir)
print("Baseline files copied successfully!")

# 2. Identify missing codes
df = pd.read_csv(crm_path, sep=";", dtype=str)
insee_col = "sirene_insee" if "sirene_insee" in df.columns else ("crm_insee" if "crm_insee" in df.columns else "insee")
postcode_col = "sirene_cp" if "sirene_cp" in df.columns else ("crm_cp" if "crm_cp" in df.columns else "postcode")

crm_insee_codes = set(df[insee_col].dropna().unique())
crm_postcodes = set(df[postcode_col].dropna().unique())

existing_insee = set()
for p in (expanded_dir / "insee").iterdir():
    if p.is_dir() and p.name.startswith("insee="):
        existing_insee.add(p.name.split("=")[1])

existing_cp = set()
for p in (expanded_dir / "cp").iterdir():
    if p.is_dir() and p.name.startswith("postcode="):
        existing_cp.add(p.name.split("=")[1])

missing_insee = sorted(list(crm_insee_codes - existing_insee))
missing_cp = sorted(list(crm_postcodes - existing_cp))

print(f"Missing INSEE codes to generate: {len(missing_insee)}")
print(f"Missing postcodes to generate: {len(missing_cp)}")

OUTPUT_SCHEMA = pa.schema([
    ("siret", pa.string()),
    ("siren", pa.string()),
    ("denomination", pa.string()),
    ("enseigne1", pa.string()),
    ("enseigne2", pa.string()),
    ("enseigne3", pa.string()),
    ("etablissementSiege", pa.bool_()),
    ("is_siege", pa.bool_()),
    ("numeroVoie", pa.string()),
    ("typeVoie", pa.string()),
    ("libelleVoie", pa.string()),
    ("complementAdresse", pa.string()),
    ("postcode", pa.string()),
    ("city", pa.string()),
    ("insee", pa.string()),
    ("cj_ul", pa.string()),
    ("etat_admin", pa.string()),
    ("last_treatment_date", pa.timestamp("us")),
    ("sigle_ul", pa.string()),
    ("denomination_ul", pa.string()),
    ("denomination_usuelle_ul", pa.string()),
    ("nom_ul", pa.string()),
    ("prenom_usuel_ul", pa.string()),
    ("pm_dirigeant_names", pa.string()),
])

def clean_adresse(adresse, cp, city):
    if not adresse:
        return ""
    addr = str(adresse).upper()
    if cp:
        addr = addr.replace(str(cp).upper(), "").strip()
    if city:
        addr = addr.replace(str(city).upper(), "").strip()
    return " ".join(addr.split())

def parse_adresse(adresse):
    if not adresse:
        return "", "", ""
    parts = adresse.split()
    if not parts:
        return "", "", ""
    
    number = ""
    if parts[0].isdigit():
        number = parts[0]
        parts = parts[1:]
    
    street_types = {"RUE", "AVENUE", "BOULEVARD", "BD", "ALLÉE", "ALLEY", "CHEMIN", "ROUTE", "PLACE", "SQUARE", "IMPASSE", "COURS", "QUAI", "VOIE", "ZONE"}
    st_type = ""
    for idx, p in enumerate(parts):
        if p.upper() in street_types:
            st_type = p.upper()
            parts = parts[idx+1:]
            break
            
    st_name = " ".join(parts)
    return number, st_type, st_name

def query_and_save_partitions(codes, code_type, column_name, root_path):
    if not codes:
        print(f"No missing codes for {code_type}.")
        return
        
    conn = sqlite3.connect(harvest_db_path)
    cursor = conn.cursor()
    
    os.makedirs(root_path, exist_ok=True)
    
    # Process in a single batch since the missing list is very small (<= 27)
    placeholders = ",".join(["?"] * len(codes))
    
    query = f"""
    SELECT 
        e.siret, e.siren, e.est_siege, e.nom_commercial, e.liste_enseignes, 
        e.adresse, e.code_postal, e.libelle_commune, e.commune, e.latitude, e.longitude,
        e.etat_administratif, e.date_creation, e.activite_principale, e.tranche_effectif, e.annee_tranche_effectif,
        ent.nom_raison_sociale, ent.nature_juridique, ent.nom_complet, ent.sigle, ent.date_mise_a_jour
    FROM etablissements e
    LEFT JOIN entreprises ent ON e.siren = ent.siren
    WHERE e.{column_name} IN ({placeholders})
    """
    
    print(f"Executing SQLite query for {len(codes)} missing {code_type} codes...")
    cursor.execute(query, codes)
    rows = cursor.fetchall()
    print(f"Fetched {len(rows)} candidate establishments.")
    
    if not rows:
        conn.close()
        return
        
    # Get dirigentes
    sirens = list({r[1] for r in rows if r[1]})
    pm_map = {}
    if sirens:
        print(f"Fetching PM dirigeants for {len(sirens)} unique SIRENs...")
        for chunk_idx in range(0, len(sirens), 900):
            chunk = sirens[chunk_idx:chunk_idx+900]
            dir_placeholders = ",".join(["?"] * len(chunk))
            dir_query = f"""
            SELECT siren, denomination
            FROM dirigeants
            WHERE siren IN ({dir_placeholders})
              AND type_dirigeant = 'personne morale'
              AND denomination IS NOT NULL
            """
            cursor.execute(dir_query, chunk)
            for siren, denom in cursor.fetchall():
                pm_map.setdefault(siren, []).append(denom)
    
    # Build pdf
    pdf_data = []
    for r in rows:
        (siret, siren, est_siege, nom_commercial, liste_enseignes, 
         adresse, code_postal, libelle_commune, commune, latitude, longitude,
         etat_administratif, date_creation, activite_principale, tranche_effectif, annee_tranche_effectif,
         nom_raison_sociale, nature_juridique, nom_complet, sigle, date_mise_a_jour) = r
         
        enseigne1, enseigne2, enseigne3 = None, None, None
        if liste_enseignes:
            try:
                ens_list = json.loads(liste_enseignes)
                if isinstance(ens_list, list):
                    if len(ens_list) > 0: enseigne1 = ens_list[0]
                    if len(ens_list) > 1: enseigne2 = ens_list[1]
                    if len(ens_list) > 2: enseigne3 = ens_list[2]
            except Exception:
                enseigne1 = liste_enseignes
        
        clean_addr = clean_adresse(adresse, code_postal, libelle_commune)
        number, st_type, st_name = parse_adresse(clean_addr)
        
        dt = None
        if date_mise_a_jour:
            try:
                dt = pd.to_datetime(date_mise_a_jour)
            except Exception:
                pass
        if dt is None:
            dt = datetime.now()
            
        pm_dirs = "|".join(pm_map.get(siren, [])) if siren in pm_map else None
        
        pdf_data.append({
            "siret": siret,
            "siren": siren,
            "denomination": nom_commercial,
            "enseigne1": enseigne1,
            "enseigne2": enseigne2,
            "enseigne3": enseigne3,
            "etablissementSiege": bool(est_siege),
            "is_siege": bool(est_siege),
            "numeroVoie": number,
            "typeVoie": st_type,
            "libelleVoie": st_name,
            "complementAdresse": None,
            "postcode": code_postal,
            "city": libelle_commune,
            "insee": commune,
            "cj_ul": nature_juridique,
            "etat_admin": etat_administratif,
            "last_treatment_date": dt,
            "sigle_ul": sigle,
            "denomination_ul": nom_raison_sociale,
            "denomination_usuelle_ul": nom_complet,
            "nom_ul": nom_raison_sociale,
            "prenom_usuel_ul": None,
            "pm_dirigeant_names": pm_dirs,
        })
        
    pdf = pd.DataFrame(pdf_data)
    if not pdf.empty:
        print(f"Writing {len(pdf)} rows to partitioned Parquet files under {root_path}...")
        table = pa.Table.from_pandas(pdf, schema=OUTPUT_SCHEMA, preserve_index=False)
        pq.write_to_dataset(table, root_path=root_path, partition_cols=[code_type], compression="ZSTD")
        print("Write complete!")
        
    conn.close()

# Generate missing CP and INSEE partitions
insee_root = expanded_dir / "insee"
cp_root = expanded_dir / "cp"

query_and_save_partitions(missing_insee, "insee", "commune", insee_root)
query_and_save_partitions(missing_cp, "postcode", "code_postal", cp_root)

print("\nDatabase expansion completed successfully!")
