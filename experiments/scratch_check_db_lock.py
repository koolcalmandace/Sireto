import sqlite3
import sys

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
print("Connecting to harvest_full.sqlite with 5s timeout...")
try:
    conn = sqlite3.connect(db_path, timeout=5.0)
    cursor = conn.cursor()
    print("Running quick SELECT...")
    cursor.execute("SELECT count(*) FROM etablissements LIMIT 1;")
    res = cursor.fetchone()
    print("Result:", res)
    conn.close()
    print("SUCCESS: Database is NOT locked!")
except Exception as e:
    print("ERROR:", e, file=sys.stderr)
    sys.exit(1)
