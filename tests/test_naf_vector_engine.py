"""
tests/test_naf_vector_engine.py — Unit Tests for INSEE NAF Semantic Vector Engine.

Verifies that commercial/trade queries are correctly projected into the
official INSEE NAF continuous vector space across diverse artisanal and commercial sectors.
"""

import pytest
from src.xgb_matcher.naf_vector_engine import NafVectorEngine, predict_naf


def test_vector_engine_singleton():
    engine1 = NafVectorEngine.get_instance()
    engine2 = NafVectorEngine.get_instance()
    assert engine1 is engine2
    assert len(engine1.naf_codes) > 50
    assert engine1.matrix.shape[0] == len(engine1.naf_codes)


@pytest.mark.parametrize(
    "query, expected_prefix",
    [
        # Artisans du bâtiment
        ("SOS PLOMBERIE DEPANNAGE 24", "43.22"),
        ("CHAUFFAGE CLIMATISATION MARTIN", "43.22"),
        ("ELECTRICIEN BATIMENT DUPONT", "43.21"),
        ("PEINTRE EN BATIMENT ET RAVALEMENT", "43.34"),
        ("CARRELEUR DU CENTRE", "43.33"),
        ("MENUISERIE BOIS ET PVC", "43.32"),
        ("CHARPENTE COUVERTURE ZINGUERIE", "43.91"),
        # Santé & Professions Libérales
        ("CABINET DENTAIRE LAFAYETTE", "86.23"),
        ("DR CHIRURGIEN DENTISTE", "86.23"),
        ("ORTHODONTISTE DU PARC", "86.23"),
        ("PHARMACIE DE LA GARE", "47.73"),
        ("CABINET KINESITHERAPIE REEDUCATION", "86.90"),
        ("CABINET MEDICAL GENERALISTE", "86.21"),
        ("CLINIQUE VETERINAIRE DES ARCADES", "75.00"),
        # Commerce & Services personnels
        ("BOULANGERIE PATISSERIE DU MOULIN", "10.71"),
        ("SALON DE COIFFURE ET BARBIER", "96.02"),
        ("INSTITUT DE BEAUTE ET SOINS", "96.02"),
        ("GARAGE MECANIQUE AUTOMOBILE CARROSSERIE", "45.20"),
        ("CAMPING DES PINS DU LAC", "55.30"),
        ("HOTEL RESTAURANT DU CENTRE", "55.10"),
        ("POMPES FUNEBRES GENERALES", "96.03"),
        ("FLEURISTE ET JARDINERIE", "47.76"),
    ],
)
def test_predict_naf_sector_accuracy(query: str, expected_prefix: str):
    preds = predict_naf(query, top_k=3, min_similarity=0.12)
    assert len(preds) > 0, f"No NAF predicted for query: {query}"
    top_code, top_score = preds[0]
    clean_top = top_code.replace(".", "")
    clean_expected = expected_prefix.replace(".", "")
    assert clean_top.startswith(clean_expected), (
        f"Query '{query}' predicted {top_code} ({top_score}), expected prefix {expected_prefix}"
    )
    assert top_score >= 0.12


def test_predict_naf_generic_empty():
    assert predict_naf("") == []
    assert predict_naf("   ") == []
    # Arbitrary non-economic gibberish should return empty or low confidence
    assert predict_naf("xyzqwk 12345", min_similarity=0.40) == []
