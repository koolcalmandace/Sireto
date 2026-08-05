import json
from pathlib import Path

model_dir = Path("models")
metas = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
if metas:
    with open(metas[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("Keys in meta:", meta.keys())
    # print ranker keys if they exist
    for k, v in meta.items():
        if "ranker" in k:
            print(f"  {k}: {v}")
