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


def test_guard4_mairie_siege_rescue():
    """Verify that a municipal town hall query (Hotel de Ville) rescues the commune's head office."""
    n = 60
    candidates = [{"siret": f"400000{i:03d}00010", "is_siege": False} for i in range(n)]

    # Slot 52: Commune de Caen - Hôtel de Ville (Siège, NAF 84.11Z)
    commune_siren = "211401187"
    mairie_siret = f"{commune_siren}00019"
    candidates[52] = {
        "siret": mairie_siret,
        "denomination": "COMMUNE DE CAEN",
        "cj_ul": "7210",
        "is_siege": True,
        "activitePrincipaleEtablissement": "84.11Z",
    }

    # Slot 10: École communale of the same commune in top-50
    candidates[10] = {
        "siret": f"{commune_siren}00450",
        "denomination": "COMMUNE DE CAEN",
        "denominationUsuelleEtablissement": "ECOLE MATERNELLE SAINT PAUL",
        "cj_ul": "7210",
        "is_siege": False,
        "activitePrincipaleEtablissement": "85.10Z",
    }

    scores = np.linspace(1.0, 0.0, n)
    crm_row = {"crm_name": "Hotel de ville Caen", "crm_adresse": "Esplanade jean marie Louvel"}

    cfg = RetentionGuardConfig(enable_mairie_siege_rescue=True)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        config=cfg,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    assert mairie_siret in retained_sirets


def test_guard4_no_hijacking_on_secondary_public_equipment():
    """Verify that an equipment query (Ecole) does NOT hijack or promote the town hall head office."""
    n = 60
    candidates = [{"siret": f"500000{i:03d}00010", "is_siege": False} for i in range(n)]

    commune_siren = "211401187"
    mairie_siret = f"{commune_siren}00019"
    # Slot 52: Mairie siège in tail
    candidates[52] = {
        "siret": mairie_siret,
        "denomination": "COMMUNE DE CAEN",
        "cj_ul": "7210",
        "is_siege": True,
        "activitePrincipaleEtablissement": "84.11Z",
        "name_jaro_max": 0.45,
    }

    # Slot 5: École communale in top-50
    ecole_siret = f"{commune_siren}00450"
    candidates[5] = {
        "siret": ecole_siret,
        "denomination": "COMMUNE DE CAEN",
        "denominationUsuelleEtablissement": "ECOLE MATERNELLE PAUL BERT",
        "cj_ul": "7210",
        "is_siege": False,
        "activitePrincipaleEtablissement": "85.10Z",
        "name_jaro_max": 0.88,
    }

    scores = np.linspace(1.0, 0.0, n)
    # Query is explicitly targeting an école
    crm_row = {"crm_name": "ECOLE PRIMAIRE PAUL BERT CAEN", "crm_adresse": "Rue Paul Bert"}

    cfg = RetentionGuardConfig(enable_mairie_siege_rescue=True)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        config=cfg,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    assert ecole_siret in retained_sirets
    # Mairie head office in tail should NOT be promoted over schools
    assert mairie_siret not in retained_sirets


def test_guard5_mono_occupant_address_rescue():
    """Verify that an exact physical address match in tail (rank < 100) is rescued if mono-occupant when enabled."""
    n = 70
    candidates = [{"siret": f"600000{i:03d}00010", "is_siege": False} for i in range(n)]

    target_siret = candidates[65]["siret"]
    candidates[65].update({
        "street_number_match": 1.0,
        "street_name_jaro": 0.92,
        "address_density": 1.0,  # Strictly single occupant
        "name_jaro_max": 0.58,   # Mild nominal signal
    })

    scores = np.linspace(1.0, 0.0, n)
    crm_row = {"crm_name": "ENTREPRISE DUPONT", "crm_adresse": "12 RUE DE LA GARE"}

    cfg = RetentionGuardConfig(enable_mono_occupant_address_rescue=True)
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        config=cfg,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    assert target_siret in retained_sirets


def test_guard5_blocked_on_multi_tenant_address():
    """Verify that an address match in tail is NOT promoted if the address has multiple tenants (density > 1)."""
    n = 70
    candidates = [{"siret": f"700000{i:03d}00010", "is_siege": False} for i in range(n)]

    target_siret = candidates[65]["siret"]
    candidates[65].update({
        "street_number_match": 1.0,
        "street_name_jaro": 0.92,
        "address_density": 12.0,  # Multi-tenant commercial zone / incubator
        "name_jaro_max": 0.58,
    })

    scores = np.linspace(1.0, 0.0, n)
    crm_row = {"crm_name": "ENTREPRISE DUPONT", "crm_adresse": "12 RUE DE LA GARE"}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    # Must NOT be promoted to prevent co-tenant noise
    assert target_siret not in retained_sirets


def test_address_match_in_top50_never_evicted():
    """
    Verify that an establishment ranked in the Top-50 by XGBoost with an exact physical
    address match (street, number, density <= 1) is NEVER evicted by retention fallback,
    even if the commercial name diverges from the legal name (e.g. HOTEL LE G vs DEVINTINA).
    """
    n = 60
    candidates = [{"siret": f"800000{i:03d}00010", "is_siege": False} for i in range(n)]

    # Slot 28 (Rank 29): Commercial hotel with exact address match and mono-occupancy
    hotel_siret = candidates[28]["siret"]
    candidates[28].update({
        "street_number_match": 1.0,
        "street_name_jaro": 1.0,
        "addr_jaro": 1.0,
        "address_density": 1.0,
        "name_jaro_max": 0.53,  # Divergent name (HOTEL LE G vs DEVINTINA)
        "name_token_overlap_max": 0.0,
        "is_siege": True,
    })

    # Slot 55: Rare token candidate in tail wanting to promote
    tail_siret = candidates[55]["siret"]
    candidates[55].update({
        "denominationUniteLegale": "SUPERCHANCEL INFORMATIQUE",
    })

    scores = np.linspace(1.0, 0.1, n)
    crm_row = {"crm_name": "HOTEL LE G SUPERCHANCEL", "crm_adresse": "1 RUE CASSINI"}
    idf_map = {"SUPERCHANCEL": 9.5}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
        name_idf_map=idf_map,
    )

    assert len(retained_sirets) == 50
    # The tail candidate is promoted
    assert tail_siret in retained_sirets
    # The physical address match in top-50 is strictly protected and NOT evicted!
    assert hotel_siret in retained_sirets


def test_upper_tier_sanctuary_ranks_1_to_35_never_evicted():
    """
    Verify Upper-Tier Sanctuary: Candidates ranked by XGBoost in slots 0 to 34 (Ranks 1 to 35)
    can NEVER be evicted to make room for tail promotions, even if their administrative denomination
    yields low name similarity (e.g. RECTORAT DE TOULOUSE at Rank 1 or POLE EMPLOI at Rank 1).
    """
    n = 60
    candidates = [{"siret": f"900000{i:03d}00010", "is_siege": False} for i in range(n)]

    # Slot 0 (Rank 1): Administrative candidate with low usage-name similarity
    rank1_siret = candidates[0]["siret"]
    candidates[0].update({
        "name_jaro_max": 0.45,
        "name_token_overlap_max": 0.0,
        "street_number_match": 0.0,
        "is_siege": False,
    })

    # Slot 55: High-priority promotion candidate in tail
    promoted_tail_siret = candidates[55]["siret"]
    candidates[55].update({
        "name_jaro_max": 0.95,
        "postcode_match": 1.0,
    })

    scores = np.linspace(1.0, 0.1, n)
    crm_row = {"crm_name": "RECTORAT DE TOULOUSE", "crm_adresse": "10 PLACE JEAN DAVID"}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    # Tail candidate promoted
    assert promoted_tail_siret in retained_sirets
    # Rank 1 candidate in upper tier sanctuary is strictly preserved at Rank 1!
    assert retained_sirets[0] == rank1_siret


def test_rule_f_near_boundary_concordance_rescue():
    """
    Verify Rule F: A near-miss candidate ranked at position 58 with local concordance
    (same commune/postcode and token overlap >= 0.50) is rescued into Top-50.
    """
    n = 65
    candidates = [{"siret": f"700000{i:03d}00010", "is_siege": False} for i in range(n)]

    # Near-miss candidate at rank 58 (slot 57)
    near_miss_siret = candidates[57]["siret"]
    candidates[57].update({
        "postcode_match": 1.0,
        "name_token_overlap_max": 0.60,
        "name_jaro_max": 0.65,
    })

    scores = np.linspace(1.0, 0.05, n)
    crm_row = {"crm_name": "CENTRE PENITENCIER", "crm_adresse": "ROUTE DE PARIS"}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    assert near_miss_siret in retained_sirets


def test_rule_f_decontracted_and_sigle_rescue():
    """
    Verify Rule F upgrade: rescues a candidate ranked at position 65 with a decontracted
    name match (e.g. L'ADAPT in CRM vs LADAPT in SIRENE) or sigle match.
    """
    n = 75
    candidates = [{"siret": f"700000{i:03d}00010", "is_siege": False} for i in range(n)]

    # Slot 64 (Rank 65): LADAPT with sigle_ul
    ladapt_siret = "77569338500285"
    candidates[64] = {
        "siret": ladapt_siret,
        "denomination_ul": "LADAPT",
        "sigle_ul": "LADAPT",
        "postcode_match": 1.0,
        "name_token_overlap_max": 0.25,  # Only 1 word out of 4 in CRM
        "name_jaro_max": 0.55,
        "is_siege": False,
    }

    scores = np.linspace(1.0, 0.05, n)
    crm_row = {"crm_name": "A0177 L'ADAPT CRF Bayeux", "crm_adresse": "RUE LOUVIERE"}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    assert ladapt_siret in retained_sirets


def test_guard1_dual_preservation_local_and_siege():
    """
    Verify Guard 1 Dual-Preservation: when a multi-site SIREN has both a local address match
    and a Siège social in Top-50, both are preserved while redundant passive clones are pruned.
    """
    n = 60
    candidates = [{"siret": f"800000{i:03d}00010", "is_siege": False} for i in range(n)]

    siren = "893906024"
    # Local address match at slot 5
    local_siret = f"{siren}00037"
    candidates[5] = {
        "siret": local_siret,
        "denomination": "ETUDE NOTARIALE",
        "is_siege": False,
        "street_number_match": 1.0,
        "street_name_jaro": 0.95,
        "name_jaro_max": 0.85,
    }

    # Siège social at slot 12
    siege_siret = f"{siren}00011"
    candidates[12] = {
        "siret": siege_siret,
        "denomination": "ETUDE NOTARIALE",
        "is_siege": True,
        "street_number_match": 0.0,
        "street_name_jaro": 0.0,
        "name_jaro_max": 0.85,
    }

    # Redundant passive clone 1 at slot 42
    clone1_siret = f"{siren}00029"
    candidates[42] = {
        "siret": clone1_siret,
        "denomination": "ETUDE NOTARIALE",
        "is_siege": False,
        "street_number_match": 0.0,
        "street_name_jaro": 0.0,
        "name_jaro_max": 0.40,
    }

    # Redundant passive clone 2 at slot 45
    clone2_siret = f"{siren}00045"
    candidates[45] = {
        "siret": clone2_siret,
        "denomination": "ETUDE NOTARIALE",
        "is_siege": False,
        "street_number_match": 0.0,
        "street_name_jaro": 0.0,
        "name_jaro_max": 0.40,
    }

    scores = np.linspace(1.0, 0.05, n)
    crm_row = {"crm_name": "ETUDE NOTARIALE ROLLAND GURRIET", "crm_adresse": "2 RUE HENRI PASTOUREAU"}

    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        crm_row=crm_row,
    )

    assert len(retained_sirets) == 50
    # Both local establishment AND Siège social MUST be preserved!
    assert local_siret in retained_sirets
    assert siege_siret in retained_sirets




