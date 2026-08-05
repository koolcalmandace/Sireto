import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Dirigeants indices:")
cursor.execute("PRAGMA index_list(dirigeants);")
for row in cursor.fetchall():
    print("  ", row)
    
conn.close()
