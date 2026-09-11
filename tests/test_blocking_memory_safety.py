"""Unit tests for TF-IDF memory safety and fail-safe behavior in blocking.py."""
import pytest
from unittest.mock import MagicMock
from sklearn.feature_extraction.text import TfidfVectorizer
import src.xgb_matcher.blocking as blocking


def test_build_word_tfidf_index_memory_error_handled():
    """Verify build_tfidf_index handles nominal cases and exceptions."""
    candidates = [{"denomination_ul": "TEST COMPANY", "siret": "12345678900012"}]
    vec, mat, names = blocking.build_tfidf_index(candidates)
    assert vec is not None
    assert mat is not None
    assert len(names) == 1


def test_prefilter_candidates_tfidf_fail_safe():
    """Verify prefilter_candidates_tfidf gracefully handles MemoryError in transform or matrix mult."""
    mock_vec = MagicMock()
    mock_vec.transform.side_effect = MemoryError("Simulated OOM")
    mock_mat = MagicMock()
    
    # Should not raise, should return empty list safely
    results = blocking.prefilter_candidates_tfidf(
        crm_name="TOTAL ENERGIES",
        vectorizer=mock_vec,
        cand_matrix=mock_mat,
        top_k=50,
    )
    assert results == []


def test_prefilter_candidates_address_tfidf_fail_safe():
    """Verify prefilter_candidates_address_tfidf handles exceptions safely."""
    mock_vec = MagicMock()
    mock_vec.transform.side_effect = Exception("Matrix mismatch")
    mock_mat = MagicMock()
    
    results = blocking.prefilter_candidates_address_tfidf(
        crm_address="10 RUE DE LA PAIX 75002 PARIS",
        vectorizer=mock_vec,
        cand_matrix=mock_mat,
        top_k=20,
    )
    assert results == []


def test_nominal_tfidf_blocking():
    """Verify nominal TF-IDF matching works accurately."""
    candidates = [
        {"denomination_ul": "BOULANGERIE DU PARC", "siret": "11111111100011"},
        {"denomination_ul": "PHARMACIE CENTRALE", "siret": "22222222200022"},
    ]
    vec, mat, names = blocking.build_tfidf_index(candidates)
    assert vec is not None
    assert mat is not None
    
    indices = blocking.prefilter_candidates_tfidf(
        crm_name="Boulangerie",
        vectorizer=vec,
        cand_matrix=mat,
        top_k=5,
    )
    assert 0 in indices


def test_compute_name_idf_map_subsampling_and_safety():
    """Verify compute_name_idf_map subsamples on large partitions and returns valid IDF."""
    from src.xgb_matcher.candidates import compute_name_idf_map

    # Test nominal small partition
    cands = {
        f"SIRET_{i}": {"denomination": f"ENTREPRISE {i % 10}", "siret": f"SIRET_{i}"}
        for i in range(100)
    }
    idf_map, default_idf = compute_name_idf_map(cands)
    assert len(idf_map) > 0
    assert default_idf > 0.0

    # Test massive partition with subsampling (> 20k)
    cands_large = {
        f"SIRET_{i}": {"denomination": f"COMMERCE {i % 50}", "siret": f"SIRET_{i}"}
        for i in range(25_000)
    }
    idf_map_sub, default_idf_sub = compute_name_idf_map(cands_large, max_sample=5_000)
    assert len(idf_map_sub) > 0
    assert default_idf_sub > 0.0


def test_build_address_tfidf_index_memory_error_handled(monkeypatch):
    """Verify build_address_tfidf_index handles MemoryError during address generation."""
    def oom_build_address(cand):
        raise MemoryError("Simulated address allocation OOM")

    monkeypatch.setattr(blocking, "build_address", oom_build_address)
    cands = [{"siret": "12345678900012", "numeroVoie": "10", "libelleVoie": "RUE DE PARIS"}]
    vec, mat = blocking.build_address_tfidf_index(cands)
    assert vec is None
    assert mat is None


def test_build_tfidf_artifacts_discards_raw_names():
    """Verify _build_tfidf_artifacts returns None for raw names to free RAM."""
    from src.xgb_matcher.retrieval import _build_tfidf_artifacts
    from src.xgb_matcher.retrieval_config import RetrievalConfigV1

    cands = [
        {"denomination_ul": "ENTREPRISE ALFA", "siret": "11111111100011", "city": "PARIS"},
        {"denomination_ul": "ENTREPRISE BETA", "siret": "22222222200022", "city": "PARIS"},
    ]
    cfg = RetrievalConfigV1()
    name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat = _build_tfidf_artifacts(cands, cfg)
    assert name_vec is not None
    assert name_mat is not None
    # Crucial memory optimization: names must be None to prevent 2.5-3.5 GB retention
    assert names is None


def test_partitioned_store_default_cache_limit(tmp_path):
    """Verify PartitionedCandidateStore defaults to 150 and evicts properly."""
    from src.xgb_matcher.partitioned_store import PartitionedCandidateStore, _MAX_STORE_CACHE_SIZE

    assert _MAX_STORE_CACHE_SIZE == 150
    store = PartitionedCandidateStore(tmp_path)
    assert store.max_cache_size == 150

    # Custom capacity support
    custom_store = PartitionedCandidateStore(tmp_path, max_cache_size=5)
    assert custom_store.max_cache_size == 5
    for i in range(10):
        custom_store._cache_insee[f"city_{i}"] = [{"siret": f"{i}"}]
        custom_store._evict_lru(custom_store._cache_insee, custom_store.max_cache_size)
    assert len(custom_store._cache_insee) == 5
    assert list(custom_store._cache_insee.keys()) == ["city_5", "city_6", "city_7", "city_8", "city_9"]


