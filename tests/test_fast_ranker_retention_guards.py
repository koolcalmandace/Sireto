"""
tests/test_fast_ranker_retention_guards.py
==========================================
Unit tests for Stage 1 Fast Ranker retention safeguards:
  - Guard 1: Signal-Gated Intra-SIREN Compaction (preserves siege & matching branches, evicts passive clones)
  - Guard 2: Address Density Dampening (evicts passive colocataires in multi-tenant buildings)
  - Guard 3: IDF Specificity Anchor & High-Name Rescue (promotes rare-token tail candidates)
  - Strict Budget Guarantee (Length == K, no budget explosion)
  - Backward Compatibility
"""

import numpy as np
import pytest

from src.xgb_matcher.retention import (
    RetentionGuardConfig,
    apply_retention_topk,
)


def test_strict_budget_preservation():
    """Ensure that apply_retention_topk always returns at most top_k candidates."""
    n = 100
    top_k = 50
    candidates = [{"siret": f"{i:09d}00010"} for i in range(n)]
    scores = np.linspace(1.0, 0.0, n)

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=top_k,
    )

    assert len(retained_indices) == top_k
    assert len(retained_sirets) == top_k
    assert len(set(retained_indices)) == top_k


def test_guard1_signal_gated_intra_siren_compaction():
    """
    Test Guard 1:
    A bank has 5 branches in top-50:
      - Branch 0: Siège social (is_siege=True)
      - Branch 1: Branch on matching CRM street (street_name_jaro=0.95)
      - Branch 2: Branch with matching enseigne (name_jaro_vs_enseigne=0.85)
      - Branch 3: Passive clone 1 (no signal)
      - Branch 4: Passive clone 2 (no signal)
    Verify:
      - Siège, Street-match, and Enseigne-match are NEVER evicted.
      - Passive clone 1 is preserved as the single allowed passive representative.
      - Passive clone 2 is evicted to make room for a promoted candidate from the tail.
    """
    siren_bank = "111222333"
    candidates = []
    # Slots 0 to 4: Bank establishments
    candidates.append({
        "siret": f"{siren_bank}00001",
        "is_siege": True,
        "street_name_jaro": 0.10,
        "name_jaro_vs_enseigne": 0.0,
    })
    candidates.append({
        "siret": f"{siren_bank}00002",
        "is_siege": False,
        "street_name_jaro": 0.95,  # Matching street!
        "name_jaro_vs_enseigne": 0.0,
    })
    candidates.append({
        "siret": f"{siren_bank}00003",
        "is_siege": False,
        "street_name_jaro": 0.10,
        "name_jaro_vs_enseigne": 0.85,  # Matching enseigne!
    })
    candidates.append({
        "siret": f"{siren_bank}00004",
        "is_siege": False,
        "street_name_jaro": 0.10,
        "name_jaro_vs_enseigne": 0.0,  # Passive clone 1
    })
    candidates.append({
        "siret": f"{siren_bank}00005",
        "is_siege": False,
        "street_name_jaro": 0.10,
        "name_jaro_vs_enseigne": 0.0,  # Passive clone 2
    })

    # Slots 5 to 49: Other generic businesses
    for i in range(5, 50):
        candidates.append({
            "siret": f"999000{i:03d}00010",
            "is_siege": False,
            "street_name_jaro": 0.10,
        })

    # Slot 50: Target candidate in tail with rare distinctive token (Chancel)
    rare_siret = "88877766600012"
    candidates.append({
        "siret": rare_siret,
        "denominationUniteLegale": "CHANCEL DISTRIBUTION",
        "is_siege": False,
    })

    # Scores: monotonic decrease
    scores = np.linspace(1.0, 0.1, len(candidates))

    crm_row = {
        "crm_name": "ETS CHANCEL",
        "crm_address": "10 RUE DE PARIS",
    }
    idf_map = {"CHANCEL": 8.5, "DISTRIBUTION": 2.1}

    cfg = RetentionGuardConfig(top_k=50)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        config=cfg,
        crm_row=crm_row,
        name_idf_map=idf_map,
    )

    # 1. Total count is strictly 50
    assert len(retained_sirets) == 50

    # 2. Rare candidate was successfully promoted into top-50
    assert rare_siret in retained_sirets

    # 3. Bank branches with signal are strictly preserved
    assert f"{siren_bank}00001" in retained_sirets  # Siège
    assert f"{siren_bank}00002" in retained_sirets  # Street match
    assert f"{siren_bank}00003" in retained_sirets  # Enseigne match
    assert f"{siren_bank}00004" in retained_sirets  # 1st passive clone preserved

    # 4. Bank passive clone 2 was evicted to free the slot
    assert f"{siren_bank}00005" not in retained_sirets


def test_guard2_address_density_dampening():
    """
    Test Guard 2:
    In a high-density domiciliation address (density=15), a candidate with
    high address match but zero name connection (name_jaro < 0.35, token_overlap=0)
    is deprioritized to allow a legitimate rare candidate from the tail to enter.
    """
    candidates = []
    # Candidate 0: High-density co-tenant (e.g. at 55 avenue de Cannes)
    cotenant_siret = "55511122200018"
    candidates.append({
        "siret": cotenant_siret,
        "denominationUniteLegale": "SCI AZUR PROPRIETES",
        "address_density": 25.0,
        "name_jaro_max": 0.12,
        "name_token_overlap_max": 0.0,
        "is_siege": False,
    })

    # Slots 1 to 49: other companies
    for i in range(1, 50):
        candidates.append({
            "siret": f"333444{i:03d}00010",
            "is_siege": True,
            "address_density": 1.0,
            "name_jaro_max": 0.70,
        })

    # Slot 50: True CRM company located in tail with rare name
    target_siret = "77788899900015"
    candidates.append({
        "siret": target_siret,
        "denominationUniteLegale": "IMEX BIO SERVICES",
        "address_density": 2.0,
        "is_siege": False,
    })

    scores = np.linspace(1.0, 0.1, len(candidates))
    crm_row = {"crm_name": "IMEX BIO", "crm_address": "55 AVENUE DE CANNES"}
    idf_map = {"IMEX": 9.2, "BIO": 4.5}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
        name_idf_map=idf_map,
    )

    assert len(retained_sirets) == 50
    # The true target is promoted into top-50
    assert target_siret in retained_sirets
    # The passive co-tenant without name match was evicted from top-50
    assert cotenant_siret not in retained_sirets


def test_guard3_idf_rarity_anchor():
    """
    Test Guard 3:
    A candidate ranked at position 55 has a rare distinct token (IDF >= 7.0)
    and is promoted into top-50.
    """
    candidates = [{"siret": f"100000{i:03d}00010", "is_siege": False} for i in range(60)]
    # Put candidate at rank 55
    rare_siret = "99988877700019"
    candidates[55] = {
        "siret": rare_siret,
        "denominationUniteLegale": "CYBE INFORMATIQUE",
        "is_siege": False,
    }

    scores = np.linspace(1.0, 0.05, 60)
    crm_row = {"crm_name": "CYBE"}
    idf_map = {"CYBE": 10.1}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
        name_idf_map=idf_map,
    )

    assert len(retained_sirets) == 50
    assert rare_siret in retained_sirets


def test_matrix_features_integration():
    """Verify that apply_retention_topk correctly consumes numpy feat_matrix."""
    n = 60
    candidates = [{"siret": f"200000{i:03d}00010"} for i in range(n)]
    feature_names = ["is_siege", "name_jaro_max", "address_density", "street_name_jaro"]
    feat_matrix = np.zeros((n, len(feature_names)), dtype=np.float32)

    # Candidate 52: Siège social in tail
    siege_siret = candidates[52]["siret"]
    # Candidate 5 is of the same SIREN
    candidates[5]["siret"] = f"{siege_siret[:9]}00099"
    feat_matrix[52, 0] = 1.0  # is_siege = 1

    scores = np.linspace(1.0, 0.0, n)
    cfg = RetentionGuardConfig(top_k=50, enable_siege_promotion=True)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        config=cfg,
        feat_matrix=feat_matrix,
        feature_names=feature_names,
    )

    assert len(retained_sirets) == 50
    assert siege_siret in retained_sirets


def test_high_name_rescue():
    """Verify that a candidate with name_jaro >= 0.85 and postcode_match == 1 in the tail is rescued."""
    n = 60
    candidates = [{"siret": f"300000{i:03d}00010", "is_siege": False} for i in range(n)]
    feature_names = ["is_siege", "name_jaro_max", "postcode_match"]
    feat_matrix = np.zeros((n, len(feature_names)), dtype=np.float32)

    target_siret = candidates[55]["siret"]
    feat_matrix[55, 1] = 0.92  # name_jaro_max = 0.92
    feat_matrix[55, 2] = 1.0   # postcode_match = 1.0

    scores = np.linspace(1.0, 0.0, n)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        feat_matrix=feat_matrix,
        feature_names=feature_names,
    )

    assert len(retained_sirets) == 50
    assert target_siret in retained_sirets

