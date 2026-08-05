import pandas as pd
from pathlib import Path

xlsx_path = Path(r"C:\Users\Kabouassi\Desktop\results_1000_sample_v42_random.xlsx")
if xlsx_path.exists():
    try:
        df = pd.read_excel(xlsx_path)
        print(f"File loaded successfully!")
        print(f"Total rows: {len(df)}")
        print("Columns:", list(df.columns))
        if "decision" in df.columns:
            print("\nDecision distribution:")
            print(df["decision"].value_counts().to_string())
        else:
            print("\nNo 'decision' column found.")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File does not exist.")
