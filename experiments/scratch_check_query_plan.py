import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

batch = ["01000"] * 100
placeholders = ",".join(["?"] * len(batch))

query = f"""
EXPLAIN QUERY PLAN
SELECT 
    e.siret, e.siren, e.est_siege, e.nom_commercial, e.liste_enseignes, 
    e.adresse, e.code_postal, e.libelle_commune, e.commune, e.latitude, e.longitude,
    e.etat_administratif, e.date_creation, e.activite_principale, e.tranche_effectif, e.annee_tranche_effectif,
    ent.nom_raison_sociale, ent.nature_juridique, ent.nom_complet, ent.sigle, ent.date_mise_a_jour
FROM etablissements e
LEFT JOIN entreprises ent ON e.siren = ent.siren
WHERE e.commune IN ({placeholders})
"""

cursor.execute(query, batch)
for row in cursor.fetchall():
    print(row)
    
conn.close()
