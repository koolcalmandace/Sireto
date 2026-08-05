import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
from src.xgb_matcher.profile import InferenceProfile

meta_file = Path("models/xgb_two_stage_meta_20260730_155642.json")
profile = InferenceProfile.from_meta(meta_file)
print(f"ranker_path: {profile.ranker_path}")
print(f"decider_path: {profile.decider_path}")
print(f"decider_path exists? {profile.decider_path.exists() if profile.decider_path else False}")
