"""
tests/test_routing_deterministic.py
===================================
Unit tests for Deterministic Natural Selection Routing (Stage 3 Pipe V7).
"""

import pytest
import numpy as np

from src.xgb_matcher.routing_deterministic import (
    DeterministicRoutingConfig,
    DeterministicRoutingResult,
    route_candidate_deterministic,
    route_from_feature_vector,
)


def test_closed_establishment_rejected():
    """Any closed establishment must be strictly rejected to REVIEW."""
    res = route_candidate_deterministic(
        score_top1=0.95,
        score_top2=0.10,
        score_gap=0.85,
        name_jaro=0.95,
        street_name_jaro=0.95,
        postcode_match=1.0,
        is_active=False,
    )
    assert res.decision == "REVIEW"
    assert res.passed_floors is False
    assert "ESTABLISHMENT_CLOSED" in res.reason


def test_low_score_rejected():
    """Score below 0.55 must be rejected to REVIEW even with large gap."""
    res = route_candidate_deterministic(
        score_top1=0.40,
        score_top2=0.10,
        score_gap=0.30,
        name_jaro=0.90,
        street_name_jaro=0.90,
        postcode_match=1.0,
        is_active=True,
    )
    assert res.decision == "REVIEW"
    assert res.passed_floors is False
    assert "SCORE_BELOW_FLOOR" in res.reason


def test_low_gap_rejected():
    """Gap below 0.10 must be rejected to REVIEW (anti-homonymy / local ambiguity)."""
    res = route_candidate_deterministic(
        score_top1=0.85,
        score_top2=0.80,
        score_gap=0.05,
        name_jaro=0.95,
        street_name_jaro=0.90,
        postcode_match=1.0,
        is_active=True,
    )
    assert res.decision == "REVIEW"
    assert res.passed_floors is False
    assert "GAP_BELOW_FLOOR" in res.reason


def test_low_name_jaro_rejected():
    """Name Jaro below 0.65 must be rejected (purely administrative match without lexical similarity)."""
    res = route_candidate_deterministic(
        score_top1=0.80,
        score_top2=0.20,
        score_gap=0.60,
        name_jaro=0.50,
        street_name_jaro=0.90,
        postcode_match=1.0,
        is_active=True,
    )
    assert res.decision == "REVIEW"
    assert res.passed_floors is False
    assert "NAME_JARO_BELOW_FLOOR" in res.reason


def test_tier_a_high_confidence():
    """High score (>=0.75), gap (>=0.15), and name_jaro (>=0.80) routes to AUTO Tier A."""
    res = route_candidate_deterministic(
        score_top1=0.88,
        score_top2=0.25,
        score_gap=0.63,
        name_jaro=0.92,
        street_name_jaro=0.70,  # Even with moderate street jaro, Tier A passes on intrinsic evidence
        postcode_match=1.0,
        is_active=True,
    )
    assert res.decision == "AUTO"
    assert res.tier == "TIER_A"
    assert res.passed_floors is True
    assert "AUTO_TIER_A" in res.reason


def test_tier_b_contextual_domination():
    """Moderate score (0.60 <= score < 0.75) with large gap (>=0.25) and strong address anchor routes to AUTO Tier B."""
    res = route_candidate_deterministic(
        score_top1=0.68,
        score_top2=0.15,
        score_gap=0.53,
        name_jaro=0.75,         # Below Tier A (0.80), but passes Floor (>=0.70)
        street_name_jaro=0.85,  # Strong street match
        postcode_match=1.0,     # Postcode match
        is_active=True,
    )
    assert res.decision == "AUTO"
    assert res.tier == "TIER_B"
    assert res.passed_floors is True
    assert "AUTO_TIER_B" in res.reason


def test_tier_b_fails_if_no_postcode():
    """Tier B must fail to REVIEW if postcode does not match."""
    res = route_candidate_deterministic(
        score_top1=0.68,
        score_top2=0.15,
        score_gap=0.53,
        name_jaro=0.75,
        street_name_jaro=0.85,
        postcode_match=0.0,
        is_active=True,
    )
    assert res.decision == "REVIEW"
    assert res.tier is None
    assert res.passed_floors is True
    assert "REVIEW_AMBIGUOUS" in res.reason


def test_route_from_feature_vector():
    """Test feature vector extraction and routing."""
    feature_names = [
        "name_jaro_max",
        "addr_jaro",
        "postcode_match",
        "street_name_jaro",
        "is_active",
    ]
    # Candidate feature values
    top1_features = np.array([0.92, 0.88, 1.0, 0.85, 1.0], dtype=np.float32)

    res = route_from_feature_vector(
        top1_features=top1_features,
        feature_names=feature_names,
        score_top1=0.82,
        score_top2=0.20,
        candidate_dict={"etat_admin": "A"},
    )
    assert res.decision == "AUTO"
    assert res.tier == "TIER_A"
    assert res.is_active is True
    assert res.name_jaro == pytest.approx(0.92)
    assert res.street_name_jaro == pytest.approx(0.85)


def test_route_from_feature_vector_closed_override():
    """Candidate dict etat_admin == 'F' must override feature vector is_active."""
    feature_names = [
        "name_jaro_max",
        "street_name_jaro",
        "postcode_match",
        "is_active",
    ]
    top1_features = np.array([0.95, 0.95, 1.0, 1.0], dtype=np.float32)

    res = route_from_feature_vector(
        top1_features=top1_features,
        feature_names=feature_names,
        score_top1=0.90,
        score_top2=0.10,
        candidate_dict={"etat_admin": "F"},
    )
    assert res.decision == "REVIEW"
    assert res.is_active is False
    assert "ESTABLISHMENT_CLOSED" in res.reason

