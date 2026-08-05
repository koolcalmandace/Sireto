import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Etablissements indices:")
cursor.execute("PRAGMA index_list(etablissements);")
for row in cursor.fetchall():
    print("  ", row)

print("\nEntreprises indices:")
cursor.execute("PRAGMA index_list(entreprises);")
for row in cursor.fetchall():
    print("  ", row)
    
conn.close()
