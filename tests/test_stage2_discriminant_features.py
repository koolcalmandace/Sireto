"""Unit tests for Stage 2 Discriminant Features (Hit@1 Maximization).

Tests:
1. civil_name_jaro: dual permutation Jaro for natural persons.
2. naf_vector_similarity: continuous NAF projection for activity discrimination.
3. delta_addr_vs_siege: address concordance differential vs group head office.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    compute_civil_name_jaro,
    compute_delta_addr_vs_siege_batch,
    compute_naf_vector_similarity,
    make_features_batch,
    make_features_from_preprocessed,
    preprocess_crm_row,
)


def test_feature_inclusion_strict():
    """Verify strict inclusion F_Stage1 ⊂ F_Stage2 with 67 features in Stage 2."""
    assert len(FAST_RANKER_FEATURE_NAMES) == 47
    assert len(FEATURE_NAMES) == 67
    for fn in FAST_RANKER_FEATURE_NAMES:
        assert fn in FEATURE_NAMES, f"Feature {fn} missing in FEATURE_NAMES"

    assert "delta_addr_vs_siege" in FEATURE_NAMES
    assert "naf_vector_similarity" in FEATURE_NAMES
    assert "civil_name_jaro" in FEATURE_NAMES


def test_civil_name_jaro_dual_permutation():
    """Verify dual permutation civil_name_jaro matches regardless of first/last name order."""
    cand = {
        "nom_ul": "DUPONT",
        "prenom_usuel_ul": "JEAN",
    }
    # CRM has "Jean Dupont"
    assert compute_civil_name_jaro("jean dupont", cand) == 1.0
    # CRM has "Dupont Jean"
    assert compute_civil_name_jaro("dupont jean", cand) == 1.0
    # CRM has "Dr Jean Dupont"
    score_dr = compute_civil_name_jaro("dr jean dupont", cand)
    assert score_dr > 0.85

    # Non-person (no nom/prenom)
    cand_sarl = {"denomination_ul": "DUPONT SERVICES"}
    assert compute_civil_name_jaro("jean dupont", cand_sarl) == 0.0


def test_naf_vector_similarity_discrimination():
    """Verify NAF vector similarity distinguishes matching vs non-matching business activity."""
    cand_boulangerie = {"activitePrincipaleEtablissement": "10.71C"}
    cand_comptable = {"activitePrincipaleEtablissement": "69.20Z"}

    sim_boul = compute_naf_vector_similarity("Boulangerie du coin", cand_boulangerie)
    sim_compt = compute_naf_vector_similarity("Boulangerie du coin", cand_comptable)

    assert sim_boul > 0.30, f"Expected > 0.30 for boulangerie, got {sim_boul}"
    assert sim_compt == 0.0, f"Expected 0.0 for comptable, got {sim_compt}"


def test_delta_addr_vs_siege_secondary_vs_siege():
    """Verify delta_addr_vs_siege promotes local secondary establishment over remote siege."""
    crm = {
        "crm_name": "Société Générale",
        "crm_adresse": "12 rue du Commerce",
        "crm_commune": "Tours",
        "crm_cp": "37000",
        "crm_insee": "37261",
    }
    crm_pre = preprocess_crm_row(crm)

    # Candidate 1: Local secondary agence at Tours
    cand_local = {
        "siret": "55212022200150",
        "is_siege": False,
        "typeVoie": "RUE",
        "libelleVoie": "DU COMMERCE",
        "numeroVoie": "12",
        "city": "Tours",
        "postcode": "37000",
        "insee": "37261",
        "denomination_ul": "SOCIETE GENERALE",
    }
    # Candidate 2: Remote Siège in Paris
    cand_siege = {
        "siret": "55212022200010",
        "is_siege": True,
        "typeVoie": "BOULEVARD",
        "libelleVoie": "HAUSSMANN",
        "numeroVoie": "29",
        "city": "Paris",
        "postcode": "75009",
        "insee": "75109",
        "denomination_ul": "SOCIETE GENERALE",
    }

    candidates = [cand_local, cand_siege]
    mat, _ = make_features_batch(crm_pre, candidates)

    delta_idx = FEATURE_NAMES.index("delta_addr_vs_siege")
    local_delta = mat[0, delta_idx]
    siege_delta = mat[1, delta_idx]

    # Local secondary should have positive delta, siege should have negative delta
    assert local_delta > 0.5, f"Expected local_delta > 0.5, got {local_delta}"
    assert siege_delta < -0.5, f"Expected siege_delta < -0.5, got {siege_delta}"
    assert local_delta > siege_delta


def test_delta_addr_vs_siege_single_candidate():
    """Verify delta_addr_vs_siege is 0.0 when there is only one candidate for the SIREN."""
    crm_pre = preprocess_crm_row({"crm_name": "Cabinet Dupont", "crm_adresse": "10 rue de la Paix"})
    cand = {
        "siret": "12345678900010",
        "is_siege": True,
        "street_name": "rue de la Paix",
        "street_number": "10",
    }
    mat, _ = make_features_batch(crm_pre, [cand])
    delta_idx = FEATURE_NAMES.index("delta_addr_vs_siege")
    assert mat[0, delta_idx] == 0.0
