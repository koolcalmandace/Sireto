import sqlite3
import time

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

insee_codes = ["01000"] * 100
placeholders = ",".join(["?"] * len(insee_codes))

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
cursor.execute(query, insee_codes)
rows = cursor.fetchall()
t1 = time.time()

print(f"Fetched {len(rows)} rows in {t1 - t0:.2f} seconds.")
conn.close()
