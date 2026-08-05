import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Just the total count - fastest query (indexed)
cursor.execute("SELECT count(*) FROM etablissements;")
total = cursor.fetchone()[0]
print(f"Total etablissements: {total:,}")

# Sample of communes to understand the format
cursor.execute("SELECT commune, code_postal, libelle_commune FROM etablissements WHERE etat_administratif='A' LIMIT 10;")
rows = cursor.fetchall()
print("\nSample rows (commune | code_postal | libelle_commune):")
for r in rows:
    print(f"  {r[0]} | {r[1]} | {r[2]}")

conn.close()
