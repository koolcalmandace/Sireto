import pytest
from pathlib import Path
from src.xgb_matcher.retrieval import _is_public_sector_query, build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.features import preprocess_crm_row


def test_extended_public_sector_prototypes():
    # Family 1: Intercommunalité
    assert _is_public_sector_query("COMMUNAUTE URBAINE DU GRAND REIMS") is True
    assert _is_public_sector_query("GRAND REIMS") is True
    assert _is_public_sector_query("COMMUNAUTE DE COMMUNES DU VAL DE SARTHE") is True

    # Family 2: Logement social / HLM / OPH
    assert _is_public_sector_query("OFFICE PUBLIC DE L HABITAT D IVRY SUR SEINE") is True
    assert _is_public_sector_query("IVRY HLM") is True
    assert _is_public_sector_query("OPH D IVRY") is True
    assert _is_public_sector_query("BAILLEUR SOCIAL AQUITAINE HABITAT") is True

    # Family 3: Emploi & Organismes Sociaux
    assert _is_public_sector_query("POLE EMPLOI SAINT DENIS") is True
    assert _is_public_sector_query("FRANCE TRAVAIL LILLE CENTRE") is True
    assert _is_public_sector_query("CAISSE D ALLOCATIONS FAMILIALES DE PARIS") is True
    assert _is_public_sector_query("CPAM DE LA GIRONDE") is True

    # Family 4: Départements & Régions
    assert _is_public_sector_query("CONSEIL DEPARTEMENTAL DU CALVADOS") is True
    assert _is_public_sector_query("DEPARTEMENT DE L AISNE") is True
    assert _is_public_sector_query("HOTEL DU DEPARTEMENT") is True
    assert _is_public_sector_query("CONSEIL REGIONAL BRETAGNE") is True

    # Family 5: Santé mentale ambulatoire & Médico-social
    assert _is_public_sector_query("CATTP AGORA") is True
    assert _is_public_sector_query("CMP ENFANTS ET ADOLESCENTS") is True
    assert _is_public_sector_query("CENTRE MEDICO PSYCHOLOGIQUE") is True
    assert _is_public_sector_query("HOPITAL DE JOUR PSYCHIATRIQUE") is True

    # Family 6: Grands corps, Enseignement Supérieur, Recherche
    assert _is_public_sector_query("CNAM BRETAGNE") is True
    assert _is_public_sector_query("UNIVERSITE DE BORDEAUX") is True
    assert _is_public_sector_query("RECTORAT DE CRETEIL") is True

    # Compounds protection
    assert _is_public_sector_query("HOTEL DE VILLE DE LYON") is True

    # Negative guards: Private businesses MUST NOT be marked public
    assert _is_public_sector_query("CARREFOUR MARKET") is False
    assert _is_public_sector_query("GARAGE RENAULT DU CENTRE") is False
    assert _is_public_sector_query("BOULANGERIE DUPONT") is False
    assert _is_public_sector_query("DOMINOS PIZZA") is False
    assert _is_public_sector_query("ORPI IMMOBILIER") is False
    assert _is_public_sector_query("PHARMACIE DE LA GARE") is False
    assert _is_public_sector_query("HOLDING FINANCIERE ALPHA SAS") is False


def test_universal_address_density_cap_10():
    config = RetrievalConfigV1(address_density_cap=10)
    assert config.address_density_cap == 10


def test_person_ul_type_casting():
    from src.xgb_matcher.naming import build_candidate_names, primary_name
    # Both int and string cj_ul must extract person names cleanly
    cand_int = {"cj_ul": 1000, "nom_ul": "LE GOADEC", "prenom_usuel_ul": "STEPHANE"}
    names_int = [n.text for n in build_candidate_names(cand_int)]
    assert "STEPHANE LE GOADEC" in names_int
    assert "LE GOADEC" in names_int
    assert primary_name(cand_int) == "STEPHANE LE GOADEC"

    cand_str = {"cj_ul": "1000", "nom_ul": "DUPONT", "prenom_usuel_ul": "JEAN"}
    names_str = [n.text for n in build_candidate_names(cand_str)]
    assert "JEAN DUPONT" in names_str
    assert "DUPONT" in names_str
    assert primary_name(cand_str) == "JEAN DUPONT"


def test_street_name_extraction_word_boundary():
    from src.xgb_matcher.features import extract_street_name_from_address
    assert extract_street_name_from_address("192 BOULEVARD DE BROU") == "BOULEVARD DE BROU"
    assert extract_street_name_from_address("15 BIS BOULEVARD DU TEMPLE") == "BOULEVARD DU TEMPLE"
    assert extract_street_name_from_address("8 QUAI DES BRUMES") == "QUAI DES BRUMES"
    assert extract_street_name_from_address("4 TOUR DU MOULIN") == "TOUR DU MOULIN"
    assert extract_street_name_from_address("10 B RUE NATIONALE") == "RUE NATIONALE"


def test_intercommunal_and_syndicats_mixtes():
    assert _is_public_sector_query("SM E-COLLECTIVITES-PEPINIERE D ENTREPRISE") is True
    assert _is_public_sector_query("SM E-COLLECTIVITES") is True
    assert _is_public_sector_query("SIVOM DE LA REGION DE BAUGE") is True
    assert _is_public_sector_query("SIVU SCOLAIRE") is True
    assert _is_public_sector_query("PETR PAYS D ARTOIS") is True
    assert _is_public_sector_query("EPE DAUDET") is True
    assert _is_public_sector_query("SDIS DE L AISNE") is True


def test_legal_corporate_stopwords():
    from src.xgb_matcher.blocking import build_tfidf_index, LEGAL_CORPORATE_STOPWORDS
    candidates = [
        {"siret": "11111111100011", "denomination": "SOCIETE DE SERVICES ET DISTRIBUTION ALPHA"},
        {"siret": "22222222200022", "denomination": "HOLDING CONSEIL ET PARTICIPATIONS BETA"},
        {"siret": "33333333300033", "denomination": "BALLANDE ASSOCIE"},
    ]
    vec, mat, names = build_tfidf_index(candidates)
    assert vec is not None
    feature_names = set(vec.get_feature_names_out())
    # Generic legal words must not be indexed as standalone distinctive features
    for stopword in ["SOCIETE", "SERVICES", "HOLDING", "PARTICIPATIONS", "CONSEIL", "DISTRIBUTION"]:
        assert stopword not in feature_names
    # Distinctive words must be preserved
    assert "ALPHA" in feature_names or any("ALPHA" in f for f in feature_names)
    assert "BALLANDE" in feature_names or any("BALLANDE" in f for f in feature_names)


def test_enseigne_priority_in_tfidf_text():
    from src.xgb_matcher.naming import candidate_tfidf_text
    cand = {
        "enseigne1": "ADAPA",
        "denomination_ul": "ASSOCIATION DEPARTEMENTALE D AIDE AUX PERSONNES DE L AIN",
    }
    tfidf_text = candidate_tfidf_text(cand)
    tokens = tfidf_text.split()
    # Commercial signboard 'ADAPA' must be boosted and appear with higher term frequency
    assert tokens.count("ADAPA") == 2
    assert "ASSOCIATION" in tokens


