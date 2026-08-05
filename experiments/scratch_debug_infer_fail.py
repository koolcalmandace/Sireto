import os
import sys
from pathlib import Path

# Set same environment variables as v4.0
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

print("Importing modules in order...")
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm

print("Importing project modules...")
from src.xgb_matcher.features import FEATURE_NAMES
from src.xgb_matcher.infer import XgbInferenceEngine
from src.xgb_matcher.profile import InferenceProfile
from src.xgb_matcher.retrieval import build_candidate_pool

print("Instantiating XGBClassifier...")
try:
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    print("Success!")
except Exception as e:
    print("Failed:", e)
    import traceback
    traceback.print_exc()
