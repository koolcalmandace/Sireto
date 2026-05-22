import pandas as pd
from pathlib import Path

def convert_csv_to_excel(csv_path: Path, xlsx_path: Path):
    if not csv_path.exists():
        print(f"Skipping: {csv_path} (does not exist)")
        return
    try:
        # Read semicolon separated CSV
        df = pd.read_csv(csv_path, sep=";")
        # Write to Excel
        df.to_excel(xlsx_path, index=False)
        print(f"Successfully converted:\n  {csv_path} -> {xlsx_path}")
    except Exception as e:
        print(f"Error converting {csv_path}: {e}")

if __name__ == "__main__":
    # Normalized reports
    convert_csv_to_excel(
        Path("data/reports_normalized/results.csv"),
        Path("data/reports_normalized/results.xlsx")
    )
    convert_csv_to_excel(
        Path("data/reports_normalized/unmatched_rows.csv"),
        Path("data/reports_normalized/unmatched_rows.xlsx")
    )
    
    # Standard reports (previous run)
    convert_csv_to_excel(
        Path("data/reports/results.csv"),
        Path("data/reports/results.xlsx")
    )
    convert_csv_to_excel(
        Path("data/reports/unmatched_rows.csv"),
        Path("data/reports/unmatched_rows.xlsx")
    )
