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
    match_spatial_sectoral_dual_anchor,
    qualify_establishment,
    run_entity_sidecar,
    run_entity_sidecar_matches,
    SidecarMatch,
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
    # Regular queries with no spatial or acronym match must NOT trigger sidecar
    candidates = [
        {"siret": "44444444400001", "denomination": "BOULANGERIE DUPONT", "activitePrincipaleEtablissement": "10.71C", "numeroVoie": "50", "libelleVoie": "RUE DE PARIS"},
        {"siret": "55555555500001", "denomination": "ORPI IMMOBILIER", "activitePrincipaleEtablissement": "68.31Z", "numeroVoie": "99", "libelleVoie": "RUE DU COMMERCE"},
    ]

    rescued = run_entity_sidecar("BOULANGERIE DUPONT", "12 Rue de Lyon", candidates)
    assert rescued == set()

    rescued_orpi = run_entity_sidecar("ORPI AGENCE CENTRALE", "1 Place de la Gare", candidates)
    assert rescued_orpi == set()


# ---------------------------------------------------------------------------
# Universal Spatial-Sectoral Dual-Anchor (SSDA) Tests
# ---------------------------------------------------------------------------

from src.xgb_matcher.sidecar import match_spatial_sectoral_dual_anchor


def test_ssda_single_tenant_exclusivity():
    candidates = [
        {
            "siret": "77777777700012",
            "denomination": "CONCEPT LINE",
            "numeroVoie": "15",
            "libelleVoie": "RUE DES FLEURS",
            "activitePrincipaleEtablissement": "46.49Z",
        }
    ]
    # Unique commercial tenant at this street address in the commune
    rescued = match_spatial_sectoral_dual_anchor("HOME SALON LITERIE", candidates, crm_address="15 Rue des Fleurs")
    assert rescued == {"77777777700012"}


def test_ssda_multi_tenant_sector_disambiguation():
    candidates = [
        {
            "siret": "88888888800011",
            "denomination": "CABINET EXPERTISE AUDIT",
            "numeroVoie": "20",
            "libelleVoie": "BOULEVARD DE LA PAIX",
            "activitePrincipaleEtablissement": "69.20Z",  # Comptabilité
        },
        {
            "siret": "88888888800022",
            "denomination": "SELARL DENTAIRE PAIX",
            "numeroVoie": "20",
            "libelleVoie": "BOULEVARD DE LA PAIX",
            "activitePrincipaleEtablissement": "86.23Z",  # Dentiste
        },
    ]

    # CRM query points to dentist: Sector anchor resolves the dentist candidate
    rescued = match_spatial_sectoral_dual_anchor("Dr Martin Chirurgien Dentiste", candidates, crm_address="20 bd de la Paix")
    assert rescued == {"88888888800022"}


def test_ssda_holding_metaclass_pass_through():
    candidates = [
        {
            "siret": "99999999900015",
            "denomination": "SO NICE HOTELS",
            "numeroVoie": "6",
            "libelleVoie": "RUE PHILIBERT",
            "activitePrincipaleEtablissement": "64.20Z",  # Holding
        }
    ]

    # Hotel banner query matched to holding alone at address via holding metaclass pass-through
    rescued = match_spatial_sectoral_dual_anchor("HOTEL DU PIN (KYRIAD)", candidates, crm_address="6 Rue Philibert")
    assert rescued == {"99999999900015"}


def test_ssda_multi_tenant_ambiguity_abort():
    # Three clothing shops at same address
    candidates = [
        {"siret": "10101010100001", "denomination": "BOUTIQUE A", "numeroVoie": "1", "libelleVoie": "CENTRE COMMERCIAL", "activitePrincipaleEtablissement": "47.71Z"},
        {"siret": "10101010100002", "denomination": "BOUTIQUE B", "numeroVoie": "1", "libelleVoie": "CENTRE COMMERCIAL", "activitePrincipaleEtablissement": "47.71Z"},
        {"siret": "10101010100003", "denomination": "BOUTIQUE C", "numeroVoie": "1", "libelleVoie": "CENTRE COMMERCIAL", "activitePrincipaleEtablissement": "47.71Z"},
    ]

    # Query for clothes shop at shopping mall: Ambiguity guard MUST abort to protect precision
    rescued = match_spatial_sectoral_dual_anchor("MAGASIN PRET A PORTER", candidates, crm_address="1 Centre Commercial")
    assert rescued == set()


# ---------------------------------------------------------------------------
# Unified 4-Field Qualification Tests (Establishment vs Holding)
# ---------------------------------------------------------------------------

def test_qualify_establishment_siege():
    cand = {
        "siret": "75290494600015",
        "siren": "752904946",
        "denomination_ul": "SO NICE HOTELS",
        "is_siege": True,
    }
    match = qualify_establishment(cand)
    assert match.siret_retenu == "75290494600015"
    assert match.type_etablissement == "SIEGE"
    assert match.siret_siege_holding == "75290494600015"
    assert match.nom_holding == "SO NICE HOTELS"


def test_qualify_establishment_secondaire():
    pool = [
        {"siret": "38491117800042", "siren": "384911178", "is_siege": True, "denomination_ul": "HOTEL CASSINI"},
        {"siret": "38491117800034", "siren": "384911178", "is_siege": False, "denomination_ul": "HOTEL CASSINI"},
    ]
    cand = pool[1]
    match = qualify_establishment(cand, pool=pool)
    assert match.siret_retenu == "38491117800034"
    assert match.type_etablissement == "SECONDAIRE"
    assert match.siret_siege_holding == "38491117800042"
    assert match.nom_holding == "HOTEL CASSINI"


def test_qualify_establishment_filiale():
    cand = {
        "siret": "38491117800034",
        "siren": "384911178",
        "denomination_ul": "SOCIETE D EXPLOITATION HOTEL CASSINI",
        "pm_dirigeant_names": "SO NICE HOTELS HOLDING",
        "is_siege": False,
    }
    match = qualify_establishment(cand)
    assert match.siret_retenu == "38491117800034"
    assert match.type_etablissement == "FILIALE"
    assert match.nom_holding == "SO NICE HOTELS HOLDING"


def test_run_entity_sidecar_matches_ssda():
    pool = [
        {
            "siret": "38491117800034",
            "siren": "384911178",
            "denomination": "HOTEL DU PIN",
            "denomination_ul": "SOCIETE D EXPLOITATION HOTEL CASSINI",
            "pm_dirigeant_names": "SO NICE HOTELS HOLDING",
            "numeroVoie": "6",
            "typeVoie": "RUE",
            "libelleVoie": "EMMANUEL PHILIBERT",
            "activitePrincipaleEtablissement": "55.10Z",
            "is_siege": False,
        },
        {
            "siret": "38491117800042",
            "siren": "384911178",
            "denomination_ul": "SOCIETE D EXPLOITATION HOTEL CASSINI",
            "numeroVoie": "152",
            "typeVoie": "BOULEVARD",
            "libelleVoie": "DES JARDINIERS",
            "is_siege": True,
        }
    ]

    matches = run_entity_sidecar_matches(
        crm_name="HOTEL DU PIN (KYRIAD)",
        crm_address="6 RUE EMMANUEL PHILIBERT",
        candidates=pool,
    )
    assert len(matches) == 1
    m = matches[0]
    assert m.siret_retenu == "38491117800034"
    assert m.type_etablissement == "FILIALE"
    assert m.nom_holding == "SO NICE HOTELS HOLDING"
    assert m.siret_siege_holding == "38491117800042"


def test_get_candidate_all_names_physical_person():
    from src.xgb_matcher.sidecar import get_candidate_all_names
    cand = {
        "siret": "81059865600066",
        "nom_ul": "DURAND DASTES",
        "prenom_usuel_ul": "CLAIRE",
        "enseigne1": "CABINET D'AVOCAT",
    }
    names = get_candidate_all_names(cand)
    assert "CLAIRE DURAND DASTES" in names
    assert "DURAND DASTES CLAIRE" in names
    assert "DURAND DASTES" in names
    assert "CABINET D'AVOCAT" in names


def test_qualify_establishment_physical_person():
    cand = {
        "siret": "81059865600066",
        "siren": "810598656",
        "nom_ul": "DURAND DASTES",
        "prenom_usuel_ul": "CLAIRE",
        "is_siege": True,
    }
    match = qualify_establishment(cand)
    assert match.siret_retenu == "81059865600066"
    assert match.type_etablissement == "SIEGE"
    assert match.nom_holding == "CLAIRE DURAND DASTES"


def test_ssda_multi_tenant_token_specificity_tie_breaking():
    # 2 automotive repair establishments at the exact same address (63 Avenue Cote Vermeille)
    candidates = [
        {
            "siret": "89175624900019",
            "nom_ul": "RODRIGUEZ",
            "prenom_usuel_ul": "CHRISTOPHER",
            "enseigne1": "GARAGE LAROQUE",
            "numeroVoie": "63",
            "typeVoie": "AVENUE",
            "libelleVoie": "COTE VERMEILLE",
            "activitePrincipaleEtablissement": "45.20A",  # Entretien et réparation de véhicules
        },
        {
            "siret": "99999999900011",
            "denomination": "SARL GARAGE COTE VERMEILLE",
            "numeroVoie": "63",
            "typeVoie": "AVENUE",
            "libelleVoie": "COTE VERMEILLE",
            "activitePrincipaleEtablissement": "45.20A",  # Same sector
        }
    ]

    # Query with distinctive proper nouns 'RODRIGUEZ' and 'LAROQUE'
    # Must break tie in favor of 89175624900019 instead of aborting
    rescued = match_spatial_sectoral_dual_anchor(
        crm_name="RODRIGUEZ GARAGE LAROQUE",
        candidates=candidates,
        crm_address="63 AVENUE COTE VERMEILLE",
    )
    assert rescued == {"89175624900019"}

