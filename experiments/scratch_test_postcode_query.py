import sqlite3
import time

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

postcodes = ['92110']
placeholders = ",".join(["?"] * len(postcodes))

print("Checking query plan...")
query = f"""
EXPLAIN QUERY PLAN
SELECT 
    e.siret, e.siren, e.est_siege, e.nom_commercial, e.liste_enseignes, 
    e.adresse, e.code_postal, e.libelle_commune, e.commune, e.latitude, e.longitude,
    e.etat_administratif, e.date_creation, e.activite_principale, e.tranche_effectif, e.annee_tranche_effectif,
    ent.nom_raison_sociale, ent.nature_juridique, ent.nom_complet, ent.sigle, ent.date_mise_a_jour
FROM etablissements e
LEFT JOIN entreprises ent ON e.siren = ent.siren
WHERE e.code_postal IN ({placeholders})
"""
cursor.execute(query, postcodes)
for row in cursor.fetchall():
    print("  ", row)

print("\nRunning actual query on 92110...")
t0 = time.time()
cursor.execute(f"""
SELECT count(*) FROM etablissements e WHERE e.code_postal = '92110'
""")
cnt = cursor.fetchone()[0]
print(f"92110 count: {cnt} rows, Time: {time.time() - t0:.2f} seconds.")

conn.close()
