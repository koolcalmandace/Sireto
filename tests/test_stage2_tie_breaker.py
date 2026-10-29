"""Unit tests for Stage 2 Tie-Breaker Engine."""
import numpy as np
import pytest

from src.xgb_matcher.features import FEATURE_NAMES
from src.xgb_matcher.tie_breaker import apply_stage2_tie_breakers, _FEAT_MAP


def _create_mock_feat(overrides: dict) -> np.ndarray:
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for k, v in overrides.items():
        if k in _FEAT_MAP:
            feat[_FEAT_MAP[k]] = float(v)
    return feat


def test_intra_siren_siege_vs_secondary_promotion():
    """Top 1 is Head Office without address match; Top 2 is local establishment with exact address."""
    cands = [
        {"siret": "25630012000050", "is_siege": 1},  # Top 1: Siège
        {"siret": "25630012000035", "is_siege": 0},  # Top 2: Local agency (same SIREN 256300120)
    ]
    scores = np.array([0.45, 0.40], dtype=np.float32)
    matrix = np.array([
        _create_mock_feat({
            "is_siege": 1.0,
            "is_active": 1.0,
            "street_number_match": 0.0,
            "exact_address_anchor": 0.0,
            "addr_jaro": 0.50,
        }),
        _create_mock_feat({
            "is_siege": 0.0,
            "is_active": 1.0,
            "street_number_match": 1.0,
            "street_name_jaro": 0.95,
            "exact_address_anchor": 1.0,
            "addr_jaro": 0.95,
            "delta_addr_vs_siege": 0.45,
        }),
    ])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is True
    assert meta["rule"] == "INTRA_SIREN_LOCAL_ESTABLISHMENT"
    assert new_order[0] == 1  # Promoted to Rank 1!
    assert meta["promoted_siret"] == "25630012000035"


def test_intra_siren_closed_secondary_never_promoted():
    """Closed establishment is never promoted over active top 1."""
    cands = [
        {"siret": "25630012000050", "is_siege": 1},
        {"siret": "25630012000035", "is_siege": 0},
    ]
    scores = np.array([0.45, 0.40], dtype=np.float32)
    matrix = np.array([
        _create_mock_feat({"is_siege": 1.0, "is_active": 1.0}),
        _create_mock_feat({"is_siege": 0.0, "is_active": 0.0, "exact_address_anchor": 1.0}),  # CLOSED!
    ])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is False
    assert new_order[0] == 0


def test_intra_siren_below_floor_never_promoted():
    """Secondary with score below min floor (<0.25) is never promoted."""
    cands = [
        {"siret": "25630012000050", "is_siege": 1},
        {"siret": "25630012000035", "is_siege": 0},
    ]
    scores = np.array([0.55, 0.18], dtype=np.float32)  # Score 0.18 is below 0.25 floor
    matrix = np.array([
        _create_mock_feat({"is_siege": 1.0, "is_active": 1.0}),
        _create_mock_feat({"is_siege": 0.0, "is_active": 1.0, "exact_address_anchor": 1.0}),
    ])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is False
    assert new_order[0] == 0


def test_inter_siren_city_suffix_promotion():
    """Inter-SIREN near-tie with city suffix match promotes specific local entity."""
    cands = [
        {"siret": "77918639400028"},  # Generic entity
        {"siret": "52290972000016"},  # Local entity with city suffix
    ]
    scores = np.array([0.62, 0.58], dtype=np.float32)  # Gap 0.04 <= 0.08
    matrix = np.array([
        _create_mock_feat({
            "is_active": 1.0,
            "name_city_suffix_match": 0.0,
            "addr_jaro": 0.60,
        }),
        _create_mock_feat({
            "is_active": 1.0,
            "name_city_suffix_match": 1.0,
            "addr_jaro": 0.85,
        }),
    ])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is True
    assert meta["rule"] == "INTER_SIREN_SPECIFICITY_MATCH"
    assert new_order[0] == 1
    assert meta["promoted_siret"] == "52290972000016"


def test_inter_siren_large_gap_blocks_promotion():
    """If ML score gap is large (>0.08), tie-breaker is NOT triggered."""
    cands = [
        {"siret": "77918639400028"},
        {"siret": "52290972000016"},
    ]
    scores = np.array([0.75, 0.55], dtype=np.float32)  # Gap 0.20 > 0.08
    matrix = np.array([
        _create_mock_feat({"is_active": 1.0, "name_city_suffix_match": 0.0}),
        _create_mock_feat({"is_active": 1.0, "name_city_suffix_match": 1.0, "addr_jaro": 0.85}),
    ])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is False
    assert new_order[0] == 0


def test_single_candidate_pass_through():
    """Single candidate pool returns order unchanged."""
    cands = [{"siret": "12345678900010"}]
    scores = np.array([0.90], dtype=np.float32)
    matrix = np.array([_create_mock_feat({"is_active": 1.0})])

    new_order, meta = apply_stage2_tie_breakers(cands, scores, matrix)
    assert meta["applied"] is False
    assert new_order == [0]
