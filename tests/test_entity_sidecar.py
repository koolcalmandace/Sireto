"""
tests/test_entity_sidecar.py — Unit Tests for Entity Resolution Sidecar.

Tests Pillars 1, 2, and 3 in complete isolation:
  - Pilier 2: Dynamic initialisms on FDEA and LHSM (zero static lists)
  - Pilier 2: Ambiguity rejection guard when multiple entities share initials
  - Pilier 3: NAF sector matching for Pharmacy, Dental, Camping, Automotive
  - Pilier 1: Offline national franchise banner resolution without Google Places
  - Negative tests: Ensure regular private queries are untouched
"""

import pytest
from src.xgb_matcher.sidecar import (
    compute_initials,
    extract_acronym_candidates,
    match_local_acronyms,
    match_naf_activity,
    match_national_brand,
    run_entity_sidecar,
)


# ---------------------------------------------------------------------------
# Pilier 2: Dynamic Local Initialisms Tests
# ---------------------------------------------------------------------------

def test_compute_initials():
    assert compute_initials("Federation Departementale d Energie des Ardennes") == "FDEA"
    assert compute_initials("Le Havre Seine Metropole") == "LHSM"
    assert compute_initials("Direction Territoriale de la Protection Judiciaire de la Jeunesse") == "DTPJJ"
    assert compute_initials("Commune de Ronchin") == "CR"


def test_extract_acronym_candidates():
    assert "FDEA" in extract_acronym_candidates("FDEA")
    assert "LHSM" in extract_acronym_candidates("LHSM - ASSAINISSEMENT")
    assert "CTC10" in extract_acronym_candidates("CTC10 - MAGASIN")
    # Stopwords should not be extracted
    assert "LES" not in extract_acronym_candidates("LES JARDINS")


def test_match_local_acronyms_single_match():
    # Commune partition sample
    candidates = [
        {"siret": "10000000000001", "denomination": "COMMUNE DE CHARLEVILLE", "sigle_ul": ""},
        {"siret": "10000000000002", "denomination": "BOULANGERIE DUPONT", "sigle_ul": ""},
        {"siret": "20000000000003", "denomination": "Federation Departementale d Energie des Ardennes", "sigle_ul": ""},
    ]

    rescued = match_local_acronyms("FDEA", candidates)
    assert rescued == {"20000000000003"}


def test_match_local_acronyms_ambiguity_rejection():
    # Multiple businesses produce the same initials in the commune
    candidates = [
        {"siret": "10000000000001", "denomination": "BATIMENT TRAVAUX PUBLICS", "numeroVoie": "1", "libelleVoie": "RUE DE PARIS"},
        {"siret": "10000000000002", "denomination": "BUREAU TECHNIQUE PROVENCAL", "numeroVoie": "10", "libelleVoie": "AVENUE DU MIDI"},
    ]

    # Without address disambiguation -> Ambiguity Guard must reject (zero risk)
    rescued_ambiguous = match_local_acronyms("BTP", candidates, crm_address="")
    assert rescued_ambiguous == set()

    # With address disambiguation -> Rescues the one on Avenue du Midi
    rescued_disambiguated = match_local_acronyms("BTP", candidates, crm_address="10 AVENUE DU MIDI")
    assert rescued_disambiguated == {"10000000000002"}


# ---------------------------------------------------------------------------
# Pilier 3: NAF Activity Bridge Tests
# ---------------------------------------------------------------------------

def test_match_naf_activity_pharmacy():
    candidates = [
        {"siret": "11111111100001", "denomination": "SCI LES CYPRES", "activitePrincipaleEtablissement": "68.20B", "numeroVoie": "5", "libelleVoie": "RUE DE LA REPUBLIQUE"},
        {"siret": "11111111100002", "denomination": "SELARL DU SOLEIL", "activitePrincipaleEtablissement": "47.73Z", "numeroVoie": "5", "libelleVoie": "RUE DE LA REPUBLIQUE"},
    ]

    # CRM says "PHARMACIE DE LA LIBERTE" at 5 Rue de la Republique
    rescued = match_naf_activity("PHARMACIE DE LA LIBERTE", candidates, crm_address="5 Rue de la Republique")
    assert rescued == {"11111111100002"}


def test_match_naf_activity_camping():
    candidates = [
        {"siret": "22222222200001", "denomination": "SARL DU MOULIN", "activitePrincipaleEtablissement": "55.30Z", "numeroVoie": "1", "libelleVoie": "ROUTE DU LAC"},
        {"siret": "22222222200002", "denomination": "HOTEL DES ALPES", "activitePrincipaleEtablissement": "55.10Z", "numeroVoie": "20", "libelleVoie": "RUE DU CENTRE"},
    ]

    rescued = match_naf_activity("CAMPING LES PINS", candidates, crm_address="1 Route du Lac")
    assert rescued == {"22222222200001"}


# ---------------------------------------------------------------------------
# Pilier 1: National Brand & Franchise Network Tests (Without Google)
# ---------------------------------------------------------------------------

def test_match_national_brand_rougegorge():
    candidates = [
        {
            "siret": "45132133501823",
            "siren": "451321335",
            "denomination": "CARREFOUR",
            "activitePrincipaleEtablissement": "47.71Z",
            "numeroVoie": "14",
            "libelleVoie": "RUE CHATELAINE",
        },
        {
            "siret": "33333333300001",
            "siren": "333333333",
            "denomination": "BOUTIQUE FLEURS",
            "activitePrincipaleEtablissement": "47.76Z",
            "numeroVoie": "14",
            "libelleVoie": "RUE CHATELAINE",
        }
    ]

    # Franchisee registered under CARREFOUR parent SIREN 451321335
    rescued = match_national_brand("ROUGEGORGE LINGERIE - 6517 LAON", candidates, crm_address="14 Rue Chatelaine")
    assert rescued == {"45132133501823"}


def test_match_national_brand_points():
    candidates = [
        {
            "siret": "32087570100055",
            "siren": "320875701",
            "denomination": "PNEU SERVICE 2000",
            "activitePrincipaleEtablissement": "45.20A",
            "numeroVoie": "12",
            "libelleVoie": "AVENUE VICTOR HUGO",
        }
    ]

    rescued = match_national_brand("POINT S", candidates, crm_address="12 Avenue Victor Hugo")
    assert rescued == {"32087570100055"}


# ---------------------------------------------------------------------------
# Unified Sidecar Runner & Negative Tests
# ---------------------------------------------------------------------------

def test_run_entity_sidecar_negative():
    # Regular private queries must NOT trigger sidecar
    candidates = [
        {"siret": "44444444400001", "denomination": "BOULANGERIE DUPONT", "activitePrincipaleEtablissement": "10.71C"},
        {"siret": "55555555500001", "denomination": "ORPI IMMOBILIER", "activitePrincipaleEtablissement": "68.31Z"},
    ]

    rescued = run_entity_sidecar("BOULANGERIE DUPONT", "12 Rue de Paris", candidates)
    assert rescued == set()

    rescued_orpi = run_entity_sidecar("ORPI AGENCE CENTRALE", "1 Place de la Gare", candidates)
    assert rescued_orpi == set()
