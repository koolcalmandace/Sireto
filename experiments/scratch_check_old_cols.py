import pandas as pd
from pathlib import Path

base_path = Path(r"C:\Users\Kabouassi\Desktop\Clean House\Project North Star\Results\sample")

for f in base_path.glob("*.xlsx"):
    try:
        df = pd.read_excel(f, nrows=2)
        print(f"File: {f.name}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error {f.name}: {e}")
