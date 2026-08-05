import os
import pandas as pd
import random
import sys

sys.path.insert(0, r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
from scripts.generate_training_samples_v5fast import load_training_data, create_siren_split

from pathlib import Path
fp = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt_v40_test.csv")
SEED = 42

if fp.exists():
    df = load_training_data(fp)
    train_df, dev_df, test_df = create_siren_split(df, SEED)
    
    print(f"Total test queries: {len(test_df)}")
    grouped = test_df.groupby("loc_key")
    print(f"Total unique test loc_keys (queries group): {len(grouped)}")
    
    # Print first 5 groups
    for idx, (loc_key, group) in enumerate(list(grouped)[:5]):
        print(f"\nGroup {idx+1} (loc_key: {loc_key}):")
        for i, row in group.iterrows():
            print(f"  crm_id={row.get('crm_id')}, crm_name={row.get('crm_name')}, postcode={row.get('postcode')}, insee={row.get('insee')}, gt_siret={row.get('ground_truth_siret')}")
else:
    print("File not found")
