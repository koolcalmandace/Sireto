import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get a sample of unique 2-character prefixes of commune
cursor.execute("SELECT DISTINCT substr(commune, 1, 2) FROM etablissements LIMIT 100;")
prefixes = [r[0] for r in cursor.fetchall() if r[0]]
print("Commune 2-digit prefixes present in SQLite:", sorted(prefixes))

conn.close()
