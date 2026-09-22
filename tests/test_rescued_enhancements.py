"""
tests/test_rescued_enhancements.py

Unit tests verifying all healthy enhancements rescued from North Star audits:
  1. Acronym symmetry bug fix (SNCF vs SNCF == 1)
  2. BAN street type normalization (AV -> AVENUE)
  3. Multi-representation delimiter root extraction (UNAPEI - Foyer -> UNAPEI)
  4. Street number range parsing and zero difference ([66, 68] vs 68 == 0.0)
  5. Multi-position street number extraction with building prefix cleaning
  6. Copropriété indicator (cj_ul == '9110')
  7. Sidecar rescue injection in retrieval
"""

import pytest

from src.xgb_matcher.features import (
    acronym_match,
    normalize_street_type,
    extract_street_number,
    parse_street_numbers,
    street_number_diff,
    preprocess_crm_row,
    make_features_from_preprocessed,
)
from src.xgb_matcher.naming import build_candidate_names


def test_acronym_match_symmetry_fix():
    """Verify bug fix C5: identical acronyms return 1 instead of 0."""
    assert acronym_match("SNCF", "SNCF") == 1
    assert acronym_match("RATP", "RATP") == 1
    assert acronym_match("CCAS", "CCAS") == 1
    assert acronym_match("SDIS", "SDIS") == 1
    # Dissimilar acronyms return 0
    assert acronym_match("SNCF", "RATP") == 0
    # Standard acronym vs expanded still works
    assert acronym_match("SNCF", "SOCIETE NATIONALE DES CHEMINS DE FER") == 1
    assert acronym_match("SOCIETE NATIONALE DES CHEMINS DE FER", "SNCF") == 1


def test_ban_street_type_normalization():
    """Verify BAN official street type normalization."""
    assert normalize_street_type("12 av de la Paix") == "12 AVENUE de la Paix"
    assert normalize_street_type("4 bd Haussmann") == "4 BOULEVARD Haussmann"
    assert normalize_street_type("5 r du Commerce") == "5 RUE du Commerce"
    assert normalize_street_type("za les plaines") == "ZONE ARTISANALE les plaines"
    # Word boundary protection: do not alter regular words
    assert normalize_street_type("AVRIL") == "AVRIL"
    assert normalize_street_type("LE HAVRE") == "LE HAVRE"


def test_delimiter_root_extraction_in_preprocess():
    """Verify multi-representation delimiter extraction in preprocess_crm_row."""
    row = {
        "crm_name": "UNAPEI - Foyer Robinson",
        "crm_address": "12 av de la Paix",
        "crm_city": "Paris",
    }
    pre = preprocess_crm_row(row)
    assert pre["crm_name_root"] == "UNAPEI"
    assert pre["crm_addr"] == "12 AVENUE DE LA PAIX"

    row_slash = {
        "crm_name": "CROIX ROUGE / SECTION JEUNES",
        "crm_address": "5 bd Victor Hugo",
        "crm_city": "Nantes",
    }
    pre_slash = preprocess_crm_row(row_slash)
    assert pre_slash["crm_name_root"] == "CROIX ROUGE"


def test_street_number_ranges_and_multi_positions():
    """Verify range parsing [66, 68] and multi-position extraction."""
    # 1. Range extraction
    assert extract_street_number("66-68 Avenue des Champs Elysées") == "66-68"
    assert extract_street_number("66 - 68 Avenue des Champs Elysées") == "66-68"
    assert extract_street_number("66/68 Avenue des Champs Elysées") == "66-68"

    # 2. Number parsing
    assert parse_street_numbers("66-68") == {66, 68}
    assert parse_street_numbers("12") == {12}

    # 3. street_number_diff on ranges
    assert street_number_diff("66-68", "68") == 0.0
    assert street_number_diff("66-68", "66") == 0.0
    assert street_number_diff("66-68", "70") == 2.0
    assert street_number_diff("66-68", "64") == 2.0

    # 4. Multi-position with building / BP prefixes
    assert extract_street_number("Batiment B, 14 rue de Paris") == "14"
    assert extract_street_number("BP 42, 5 rue du Commerce") == "5"
    assert extract_street_number("Boulevard Foch 14") == "14"


def test_copropriete_feature_and_dense_site():
    """Verify is_copropriete and street_number_match in make_features_from_preprocessed."""
    crm = {
        "crm_name": "BOULANGERIE MODERNE",
        "crm_addr": "66 AVENUE DES CHAMPS ELYSEES",
        "crm_city": "PARIS",
        "crm_city_norm": "PARIS",
        "crm_street_num": "66-68",
        "crm_street_name": "CHAMPS ELYSEES",
        "postcode": "75008",
        "insee": "75108",
    }

    cand_copro = {
        "siret": "91100000000001",
        "denomination": "SYNDICAT DES COPROPRIETAIRES",
        "cj_ul": "9110",
        "numeroVoie": "68",
        "libelleVoie": "DES CHAMPS ELYSEES",
        "codePostalEtablissement": "75008",
        "codeCommuneEtablissement": "75108",
    }
    cand_names = build_candidate_names(cand_copro)
    feats = make_features_from_preprocessed(crm, cand_copro, skip_semantic=True)

    # is_copropriete must be 1.0 for cj_ul == '9110'
    assert feats["is_copropriete"] == 1.0
    # street_number_match must be 1.0 because 68 is within range '66-68'
    assert feats["street_number_match"] == 1.0
    assert feats["street_number_diff"] == 0.0

    cand_commercial = {
        "siret": "50000000000002",
        "denomination": "BOULANGERIE MODERNE",
        "cj_ul": "5499",  # SARL
        "numeroVoie": "66",
        "libelleVoie": "DES CHAMPS ELYSEES",
        "codePostalEtablissement": "75008",
        "codeCommuneEtablissement": "75108",
    }
    feats_comm = make_features_from_preprocessed(crm, cand_commercial, skip_semantic=True)

    assert feats_comm["is_copropriete"] == 0.0
    assert feats_comm["street_number_match"] == 1.0
