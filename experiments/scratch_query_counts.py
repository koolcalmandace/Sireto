import sqlite3

db_path = r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\harvest_full.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

insee_list = ['01053', '02303', '02340', '02383', '02408', '02691', '02722', '03013', '03023', '03190', '03286', '03310', '03321', '06004', '06018', '06023', '06027', '06029', '06030', '06048']

for code in insee_list:
    cursor.execute("SELECT count(*) FROM etablissements WHERE commune = ?;", (code,))
    cnt = cursor.fetchone()[0]
    print(f"Commune {code}: {cnt} establishments")
    
conn.close()
