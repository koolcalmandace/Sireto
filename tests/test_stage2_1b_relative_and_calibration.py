"""Unit tests for Stage 2 1B within-query relative features and SigmoidCalibrator."""

from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.xgb_matcher.calibration import SigmoidCalibrator, load_calibrator
from src.xgb_matcher.features import (
    FAST_RANKER_FEATURE_NAMES,
    FEATURE_NAMES,
    populate_query_relative_features,
)


class _DummyEstimator:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(X[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def test_feature_dimensions_preserved():
    """Stage 1 Fast Ranker must stay at 47 features; Stage 2 expands to 72."""
    assert len(FAST_RANKER_FEATURE_NAMES) == 47
    assert len(FEATURE_NAMES) == 72
    assert set(FAST_RANKER_FEATURE_NAMES).issubset(set(FEATURE_NAMES))


def test_populate_query_relative_features_intra_siren_branch_vs_siege():
    """Local branch with better street must get is_best_addr_in_siren=1.0 over Siege."""
    mat = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)

    idx_addr = FEATURE_NAMES.index("addr_jaro")
    idx_name = FEATURE_NAMES.index("name_jaro_max")
    idx_dist = FEATURE_NAMES.index("spatial_distance_km")
    idx_anchor = FEATURE_NAMES.index("exact_address_anchor")

    # Cand 0: Siege (same SIREN 111111111), weak address, far GPS
    mat[0, idx_addr] = 0.50
    mat[0, idx_name] = 0.95
    mat[0, idx_dist] = 12.0
    mat[0, idx_anchor] = 0.0

    # Cand 1: Local branch (same SIREN 111111111), exact address, 50m GPS
    mat[1, idx_addr] = 0.96
    mat[1, idx_name] = 0.95
    mat[1, idx_dist] = 0.05
    mat[1, idx_anchor] = 1.0

    # Cand 2: Rival (different SIREN 222222222), medium address
    mat[2, idx_addr] = 0.70
    mat[2, idx_name] = 0.60
    mat[2, idx_dist] = 2.0
    mat[2, idx_anchor] = 0.0

    cands = [
        {"siret": "11111111100010"},
        {"siret": "11111111100028"},
        {"siret": "22222222200015"},
    ]

    populate_query_relative_features(mat, cands)

    assert mat[1, FEATURE_NAMES.index("rel_addr_to_max")] == np.float32(0.0)
    assert mat[0, FEATURE_NAMES.index("rel_addr_to_max")] < -0.40
    assert mat[1, FEATURE_NAMES.index("rel_spatial_to_min")] == np.float32(0.0)
    assert mat[0, FEATURE_NAMES.index("rel_spatial_to_min")] > 11.0

    assert mat[0, FEATURE_NAMES.index("is_best_addr_in_siren")] == 0.0
    assert mat[1, FEATURE_NAMES.index("is_best_addr_in_siren")] == 1.0
    assert mat[2, FEATURE_NAMES.index("is_best_addr_in_siren")] == 1.0

    assert mat[0, FEATURE_NAMES.index("siren_count_in_top50")] == 2.0
    assert mat[1, FEATURE_NAMES.index("siren_count_in_top50")] == 2.0
    assert mat[2, FEATURE_NAMES.index("siren_count_in_top50")] == 1.0


def test_sigmoid_calibrator_monotonic_and_picklable(tmp_path: Path):
    """SigmoidCalibrator produces smooth monotonic probabilities and round-trips via load_calibrator."""
    X = np.linspace(0.1, 0.9, 20).reshape(-1, 1)
    y = np.array([0] * 10 + [1] * 10)
    lr = LogisticRegression().fit(X, y)
    cal = SigmoidCalibrator(_DummyEstimator(), lr)

    probs = cal.predict_proba(np.array([[0.2], [0.5], [0.8]]))[:, 1]
    assert probs[0] < probs[1] < probs[2]
    assert 0.0 < probs[0] and probs[2] < 1.0

    pkl_path = tmp_path / "sigmoid_cal.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(cal, f)

    loaded = load_calibrator(pkl_path)
    probs_loaded = loaded.predict_proba(np.array([[0.2], [0.5], [0.8]]))[:, 1]
    np.testing.assert_allclose(probs, probs_loaded, rtol=1e-6)
