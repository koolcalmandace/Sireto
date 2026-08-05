import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Total establishments
cursor.execute("SELECT count(*) FROM etablissements;")
total = cursor.fetchone()[0]

# Active only
cursor.execute("SELECT count(*) FROM etablissements WHERE etat_administratif = 'A';")
active = cursor.fetchone()[0]

# Distinct communes covered
cursor.execute("SELECT count(DISTINCT commune) FROM etablissements;")
communes = cursor.fetchone()[0]

# Distinct code_postal covered
cursor.execute("SELECT count(DISTINCT code_postal) FROM etablissements;")
postcodes = cursor.fetchone()[0]

print(f"Total etablissements:          {total:,}")
print(f"Active only (etat='A'):        {active:,}")
print(f"Distinct communes in SQLite:   {communes:,}")
print(f"Distinct code_postal in SQLite:{postcodes:,}")

conn.close()
