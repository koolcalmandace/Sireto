from pathlib import Path

p = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\candidates_v7_expanded")
if not p.exists():
    print("Directory does not exist yet.")
else:
    insee_files = list((p / "insee").rglob("*.parquet"))
    cp_files = list((p / "cp").rglob("*.parquet"))
    print(f"Insee partitions generated: {len(insee_files)}")
    print(f"CP partitions generated: {len(cp_files)}")
