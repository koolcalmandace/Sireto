import pandas as pd
from pathlib import Path

import sys

results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/reports/results.csv")
unmatched_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/reports/unmatched_rows.csv")

if not results_path.exists():
    print(f"Error: {results_path} does not exist!")
    exit(1)

# Read results.csv using semicolon separator
df = pd.read_csv(results_path, sep=";")

# Filter for NO_MATCH
df_unmatched = df[df["status"] == "NO_MATCH"]

# Save to unmatched_rows.csv
df_unmatched.to_csv(unmatched_path, index=False, sep=";")

print(f"Successfully extracted {len(df_unmatched)} unmatched rows out of {len(df)} total rows.")
print(f"Saved unmatched rows to {unmatched_path}")

# Print first 15 unmatched rows as an example
print("\n--- Highlight of Unmatched Rows (First 15) ---")
cols_to_show = ["crm_id", "crm_name", "normalized_name", "normalized_address", "reason"]
for idx, row in df_unmatched[cols_to_show].head(15).iterrows():
    print(f"CRM ID: {row['crm_id']} | Name: {row['crm_name']} | Normalized: {row['normalized_name']} | Address: {row['normalized_address']}")
    print(f"  Reason: {row['reason']}")
    print("-" * 50)
