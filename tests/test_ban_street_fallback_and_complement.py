"""
test_ban_street_fallback_and_complement.py
==========================================
Unit tests verifying:
  1. BAN street fallback resolution for adjacent/boundary communes (SpatialResolver).
  2. Candidate address enrichment via cleaned complementAdresse (build_address_for_retrieval).
  3. Feature immutability: build_address() remains unchanged for XGBoost ML features.
"""

import pytest
from src.xgb_matcher.spatial import SpatialResolver
from src.xgb_matcher.features import build_address, build_address_for_retrieval


def test_build_address_retrieval_complement_enrichment():
    cand_with_comp = {
        "numeroVoie": "12",
        "typeVoie": "RUE",
        "libelleVoie": "DE LA PAIX",
        "complementAdresse": "PARC D ACTIVITES DU MOULIN BATIMENT B BP 4001 CEDEX 02",
    }
    
    # 1. build_address (ML features) must remain untouched
    base_addr = build_address(cand_with_comp)
    assert base_addr == "12 RUE DE LA PAIX"
    
    # 2. build_address_for_retrieval (TF-IDF) must include cleaned complement
    retrieval_addr = build_address_for_retrieval(cand_with_comp)
    assert "PARC D ACTIVITES DU MOULIN" in retrieval_addr
    assert "BP" not in retrieval_addr
    assert "CEDEX" not in retrieval_addr
    assert "12 RUE DE LA PAIX" in retrieval_addr


def test_ban_street_fallback_resolver():
    sp = SpatialResolver()
    if not sp.is_available:
        pytest.skip("BAN database not available")

    # Street 'CADUCEE' in dept 34 is in Montpellier (34172)
    insee_34 = sp.resolve_commune_by_street("34", "RUE DU CADUCEE")
    assert insee_34 == "34172"

    # Street 'TAFFIGNON' in dept 69 is in Sainte-Foy-les-Lyon (69202)
    insee_69 = sp.resolve_commune_by_street("69", "ALLEE DE TAFFIGNON")
    assert insee_69 == "69202"
