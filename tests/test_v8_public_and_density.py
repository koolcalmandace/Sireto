import pytest
from src.xgb_matcher.retrieval import _is_public_sector_query, build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.features import preprocess_crm_row
from pathlib import Path


def test_is_public_sector_query():
    # Public / Municipal queries should be detected
    assert _is_public_sector_query("ECOLE ELEMENTAIRE LEON BLUM") is True
    assert _is_public_sector_query("MAIRIE DU PLESSIS PATE") is True
    assert _is_public_sector_query("GYMNASE MUNICIPAL") is True
    assert _is_public_sector_query("CENTRE COMMUNAL D'ACTION SOCIALE") is True
    assert _is_public_sector_query("COLLEGE JEAN MOULIN") is True

    # Commercial / Private queries should NOT be detected as public
    assert _is_public_sector_query("CARREFOUR MARKET") is False
    assert _is_public_sector_query("BOULANGERIE DUPONT") is False
    assert _is_public_sector_query("GARAGE DU CENTRE") is False
    assert _is_public_sector_query("PHARMACIE DE LA PLACE") is False


def test_public_expansion_and_address_density():
    store = PartitionedCandidateStore(Path("data/candidates_v8_national"))
    config = RetrievalConfigV1(
        rescue_public_expansion=True,
        address_density_cap=5,
        prefilter_k=500,
    )

    # Test query: Ecole elementaire in Plessis-Pate (91494)
    crm_row = {
        "crm_id": "TEST_PUBLIC_1",
        "crm_name": "ECOLE ELEMENTAIRE LEON BLUM",
        "crm_insee": "91494",
        "crm_cp": "91220",
        "crm_adresse": "15 AVENUE GILBERT FERGANT",
    }
    crm_pre = preprocess_crm_row(crm_row)
    tfidf_cache = {}

    res = build_candidate_pool(
        store=store,
        crm_row=crm_row,
        crm_pre=crm_pre,
        config=config,
        tfidf_cache=tfidf_cache,
        compute_idf_map=False,
    )

    # Base pool should be loaded
    assert res.pool_sizes["base"] > 0
    # Secondary municipal establishments should be in prefilter
    public_sirets = [c.get("siret") for c in res.candidates if str(c.get("cj_ul") or "").startswith("7")]
    assert len(public_sirets) > 0
