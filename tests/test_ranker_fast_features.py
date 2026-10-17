from __future__ import annotations

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
)
from pathlib import Path

from src.xgb_matcher.profile import InferenceProfile


def test_fast_ranker_features_subset_and_order() -> None:
    assert FAST_RANKER_FEATURE_NAMES
    assert len(set(FAST_RANKER_FEATURE_NAMES)) == len(FAST_RANKER_FEATURE_NAMES)
    assert all(name in FEATURE_NAMES for name in FAST_RANKER_FEATURE_NAMES)
    assert not set(FAST_RANKER_FEATURE_NAMES) & set(SEMANTIC_FEATURE_NAMES)

    positions = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]
    assert positions == sorted(positions)

    for name in ("name_jaro_max", "addr_jaro", "geo_exact_match"):
        assert name in FAST_RANKER_FEATURE_NAMES


def test_profile_stage1_feature_order_prefers_fast() -> None:
    profile = InferenceProfile(
        feature_order=["full_a", "full_b"],
        ranker_feature_order=["ranker_a"],
        ranker_fast_feature_order=["fast_a"],
        ranker_fast_path=Path("models/xgbranker_fast.json"),
        use_ranker_fast=True,
    )
    assert profile.get_stage1_feature_order() == ["fast_a"]

    profile.use_ranker_fast = False
    assert profile.get_stage1_feature_order() == ["ranker_a"]

    profile.ranker_feature_order = []
    assert profile.get_stage1_feature_order() == ["full_a", "full_b"]


def test_v9_fast_ranker_features() -> None:
    from src.xgb_matcher.features import make_features_from_preprocessed, preprocess_crm_row

    for expected_feat in ("exact_address_anchor", "is_commune_authority", "acronym_mutual_token_match", "is_association"):
        assert expected_feat in FAST_RANKER_FEATURE_NAMES

    # Test 1: is_commune_authority (CJ 7210 in same commune)
    crm = preprocess_crm_row({
        "crm_name": "ECOLE PRIMAIRE JEAN JAURES",
        "crm_adresse": "10 RUE DE LA PAIX",
        "insee": "75056",
        "postcode": "75001",
    })
    cand_commune = {
        "denomination": "COMMUNE DE PARIS",
        "cj_ul": "7210",
        "insee": "75056",
        "codeCommuneEtablissement": "75056",
        "numeroVoie": "10",
        "typeVoie": "RUE",
        "libelleVoie": "DE LA PAIX",
    }
    feats = make_features_from_preprocessed(crm, cand_commune, skip_semantic=True)
    assert feats["is_commune_authority"] == 1.0
    assert feats["exact_address_anchor"] == 1.0

    # Test 2: candidate in another commune
    cand_other_commune = {
        "denomination": "COMMUNE DE LYON",
        "cj_ul": "7210",
        "insee": "69123",
        "codeCommuneEtablissement": "69123",
    }
    feats_other = make_features_from_preprocessed(crm, cand_other_commune, skip_semantic=True)
    assert feats_other["is_commune_authority"] == 0.0

    # Test 3: acronym_mutual_token_match
    crm_acronym = preprocess_crm_row({
        "crm_name": "LE NOBLE AGE",
        "crm_adresse": "10 RUE DE LA PAIX",
        "insee": "75056",
    })
    cand_holding = {
        "denomination": "LNA SANTE",
        "cj_ul": "5710",
        "insee": "75056",
    }
    feats_acronym = make_features_from_preprocessed(crm_acronym, cand_holding, skip_semantic=True)
    assert feats_acronym["acronym_mutual_token_match"] == 1.0
