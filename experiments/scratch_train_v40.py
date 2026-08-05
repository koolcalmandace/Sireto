import subprocess
import os
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== Starting Version 4.0 Pipeline Training and Test Run ===")

# Paths
gt_csv = r"data/crm_ok_gt.csv"
test_gt_csv = r"data/crm_ok_gt_v40_test.csv"
samples_ranker = r"data/samples_v40_ranker.parquet"
samples_decider = r"data/samples_v40_decider.parquet"
python_exe = r".\venv\Scripts\python.exe"

# 1. Create a smaller subset of GT (250 records) for fast training verification
if not os.path.exists(test_gt_csv):
    print(f"Creating {test_gt_csv} from {gt_csv}...")
    try:
        try:
            df = pd.read_csv(gt_csv, sep=";", dtype=str)
        except Exception:
            df = pd.read_csv(gt_csv, sep=None, engine="python", dtype=str)
        
        df_subset = df.iloc[:250]
        df_subset.to_csv(test_gt_csv, index=False, sep=";")
        print(f"Saved {len(df_subset)} records to {test_gt_csv}.")
    except Exception as e:
        print(f"Error creating subset: {e}")
        sys.exit(1)
else:
    print(f"{test_gt_csv} already exists.")

# 2. Run Step 1: Generate Ranker Training Samples
print("\n[STEP 1/4] Generating Ranker training samples (V4.0 Jaccard features)...")
cmd_gen_ranker = [
    python_exe,
    "scripts/generate_training_samples_v5fast.py",
    "--mode=ranker",
    "--training-csv", test_gt_csv,
    "--output", samples_ranker
]
env = os.environ.copy()
env["PYTHONPATH"] = "src;."
env["XGB_SEMANTIC_ENABLED"] = "0"
env["XGB_SAMPLE_WORKERS"] = "6"

res = subprocess.run(cmd_gen_ranker, env=env)
if res.returncode != 0:
    print(f"Ranker sample generation failed with code {res.returncode}")
    sys.exit(1)
print("Ranker training samples generated successfully.")

# 3. Run Step 2: Train Ranker Model
print("\n[STEP 2/4] Training Ranker model (Stage 1)...")
cmd_train_ranker = [
    python_exe,
    "scripts/train_xgb_ranker.py",
    "--samples", samples_ranker
]
res = subprocess.run(cmd_train_ranker, env=env)
if res.returncode != 0:
    print(f"Ranker model training failed with code {res.returncode}")
    sys.exit(1)
print("Ranker model trained successfully.")

# 4. Run Step 3: Generate Decider Training Samples (requires trained Ranker)
print("\n[STEP 3/4] Generating Decider training samples...")
env["XGB_SEMANTIC_ENABLED"] = "1"
cmd_gen_decider = [
    python_exe,
    "scripts/generate_training_samples_v5fast.py",
    "--mode=decider",
    "--training-csv", test_gt_csv,
    "--output", samples_decider
]
res = subprocess.run(cmd_gen_decider, env=env)
if res.returncode != 0:
    print(f"Decider sample generation failed with code {res.returncode}")
    sys.exit(1)
print("Decider training samples generated successfully.")

# 5. Run Step 4: Train Decider Model
print("\n[STEP 4/4] Training Decider model (Stage 2)...")
cmd_train_decider = [
    python_exe,
    "scripts/train_xgb_decider.py",
    "--samples", samples_decider
]
res = subprocess.run(cmd_train_decider, env=env)
if res.returncode != 0:
    print(f"Decider model training failed with code {res.returncode}")
    sys.exit(1)
print("Decider model trained successfully.")

print("\n=== Version 4.0 Pipeline Models Trained Successfully ===")
