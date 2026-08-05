from pathlib import Path

p = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_all")
print("Exists:", p.exists())

insee_dir = p / "insee"
cp_dir = p / "cp"

print("Insee exists:", insee_dir.exists())
print("CP exists:", cp_dir.exists())

# List first 10 items in insee and cp
if insee_dir.exists():
    items = list(insee_dir.glob("insee=*"))
    print("Insee partitions count:", len(items))
    print("First 10 Insee partitions:", [x.name for x in items[:10]])

if cp_dir.exists():
    items = list(cp_dir.glob("postcode=*"))
    print("CP partitions count:", len(items))
    print("First 10 CP partitions:", [x.name for x in items[:10]])
