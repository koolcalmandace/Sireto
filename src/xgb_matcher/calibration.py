import pickle
from typing import Any
import numpy as np

class IsotonicCalibrator:
    """Wrapper for calibrated probabilities using Isotonic Regression."""
    def __init__(self, base_estimator: Any, iso_reg: Any) -> None:
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_estimator, "predict_proba"):
            proba = self.base_estimator.predict_proba(X)[:, 1]
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            proba = self.base_estimator.predict(dmat)
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SigmoidCalibrator:
    """Wrapper for smooth sigmoid (Platt) calibration of a base XGBoost estimator."""

    def __init__(self, base_estimator: Any, lr: Any) -> None:
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_estimator, "predict_proba"):
            proba = self.base_estimator.predict_proba(X)[:, 1]
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            proba = self.base_estimator.predict(dmat)
        return self.lr.predict_proba(np.asarray(proba, dtype=np.float64).reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_calibrator(path):
    """Robustly load a calibrator pickle, handling module path changes."""
    # Shim to redirect old script module path to new location
    # This allows unpickling objects that were saved as 'scripts.route_xgb_results.IsotonicCalibrator'
    # or '__main__.IsotonicCalibrator'
    
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'IsotonicCalibrator':
                return IsotonicCalibrator
            if name == 'SigmoidCalibrator':
                return SigmoidCalibrator
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return Unpickler(f).load()
