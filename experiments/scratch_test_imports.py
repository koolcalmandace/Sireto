import sys
from pathlib import Path

_PROJECT_ROOT = Path(r"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto")
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

print("Importing sklearn...")
try:
    import sklearn
    print("sklearn version:", sklearn.__version__)
except Exception as e:
    print("Failed to import sklearn:", e)
    import traceback
    traceback.print_exc()

print("Importing xgboost...")
import xgboost as xgb
print("xgboost version:", xgb.__version__)

print("Instantiating XGBClassifier...")
try:
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    print("Success instantiating XGBClassifier!")
except Exception as e:
    print("Failed to instantiate XGBClassifier:", e)
    import traceback
    traceback.print_exc()
