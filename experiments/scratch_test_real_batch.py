import pandas as pd
import sqlite3
import time

crm_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv"
df = pd.read_csv(crm_path, sep=";", dtype=str)
insee_col = "sirene_insee" if "sirene_insee" in df.columns else ("crm_insee" if "crm_insee" in df.columns else "insee")
insee_codes = sorted(list(df[insee_col].dropna().unique()))

print("Actual first 10 INSEE codes:", insee_codes[:10])

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

batch = insee_codes[:100]
placeholders = ",".join(["?"] * len(batch))

query = f"""
SELECT 
    e.siret, e.siren, e.est_siege, e.nom_commercial, e.liste_enseignes, 
    e.adresse, e.code_postal, e.libelle_commune, e.commune, e.latitude, e.longitude,
    e.etat_administratif, e.date_creation, e.activite_principale, e.tranche_effectif, e.annee_tranche_effectif,
    ent.nom_raison_sociale, ent.nature_juridique, ent.nom_complet, ent.sigle, ent.date_mise_a_jour
FROM etablissements e
LEFT JOIN entreprises ent ON e.siren = ent.siren
WHERE e.commune IN ({placeholders})
"""

t0 = time.time()
cursor.execute(query, batch)
rows = cursor.fetchall()
t1 = time.time()

print(f"Fetched {len(rows)} rows in {t1 - t0:.2f} seconds.")

# Profile the parsing and cleaning logic for these rows
t2 = time.time()
sirens = list({r[1] for r in rows if r[1]})
pm_map = {}
if sirens:
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

print(f"Loaded dirigeants for {len(sirens)} sirens in {time.time() - t2:.2f} seconds.")
conn.close()
