import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check how many records exist for dept 92
cursor.execute("SELECT count(*) FROM etablissements WHERE commune LIKE '92%';")
cnt_92 = cursor.fetchone()[0]
print(f"Commune LIKE '92%' count: {cnt_92}")

# Sample actual commune codes for dept 92
cursor.execute("SELECT DISTINCT commune FROM etablissements WHERE commune LIKE '92%' LIMIT 20;")
communes = [r[0] for r in cursor.fetchall()]
print("Sample commune codes for dept 92:", sorted(communes))

# Sample postcode values for dept 92 communes
if communes:
    placeholders = ",".join(["?"] * len(communes))
    cursor.execute(f"SELECT DISTINCT code_postal FROM etablissements WHERE commune IN ({placeholders}) LIMIT 20;", communes)
    postcodes = [r[0] for r in cursor.fetchall()]
    print("Sample postcodes for dept 92:", sorted(postcodes))

conn.close()
