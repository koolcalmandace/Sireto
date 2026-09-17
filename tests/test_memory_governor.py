"""
Tests for MemoryGovernor and memory safety features.
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.memory_governor import MemoryGovernor
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import _check_siret_in_list, inject_gt_if_missing, CandidatePoolResult


def test_memory_governor_rss_reading():
    """Verify MemoryGovernor reads non-negative process RSS."""
    gov = MemoryGovernor(target_max_gb=4.0, critical_max_gb=4.5)
    rss_bytes = gov.get_rss_bytes()
    rss_mb = gov.get_rss_mb()
    rss_gb = gov.get_rss_gb()

    assert rss_bytes > 0
    assert rss_mb > 0
    assert rss_gb > 0
    assert abs(rss_mb - (rss_bytes / (1024 * 1024))) < 0.01


def test_memory_governor_check_and_release():
    """Verify check_and_release flushes caches when forced or when threshold is low."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        store = PartitionedCandidateStore(tmp_path)

        # Populate dummy store cache
        store._cache_insee["75056"] = [{"siret": "12345678900014"}]
        store._cache_cp["75001"] = [{"siret": "12345678900014"}]
        store._cache_cp_insee["75001|75056"] = [{"siret": "12345678900014"}]
        store._cache_insee_count["75056"] = 100

        tfidf_cache = OrderedDict()
        tfidf_cache["key1"] = "dummy_data"

        assert len(store._cache_insee) == 1
        assert len(tfidf_cache) == 1

        # High ceiling -> should not release
        gov_high = MemoryGovernor(target_max_gb=1000.0, critical_max_gb=2000.0, check_interval=1)
        released = gov_high.check_and_release(store=store, tfidf_cache=tfidf_cache, force=False)
        assert not released
        assert len(store._cache_insee) == 1
        assert len(tfidf_cache) == 1

        # Force release -> must purge caches
        released_forced = gov_high.check_and_release(store=store, tfidf_cache=tfidf_cache, force=True)
        assert released_forced
        assert len(store._cache_insee) == 0
        assert len(store._cache_cp) == 0
        assert len(store._cache_cp_insee) == 0
        assert len(store._cache_insee_count) == 0
        assert len(tfidf_cache) == 0


def test_store_clear_all_caches():
    """Verify clear_all_caches on PartitionedCandidateStore empties all cache dicts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        store = PartitionedCandidateStore(tmp_path)

        store._cache_insee["33063"] = [{"siret": "33333333300011"}]
        store._cache_cp["33000"] = [{"siret": "33333333300011"}]
        store._cache_cp_insee["33000|33063"] = [{"siret": "33333333300011"}]
        store._cache_dept["33"] = [{"siret": "33333333300011"}]
        store._cache_insee_count["33063"] = 42

        store.clear_all_caches()

        assert len(store._cache_insee) == 0
        assert len(store._cache_cp) == 0
        assert len(store._cache_cp_insee) == 0
        assert len(store._cache_dept) == 0
        assert len(store._cache_insee_count) == 0


def test_check_siret_in_list_lazy():
    """Verify _check_siret_in_list works accurately with 0 allocation."""
    candidates = [
        {"siret": "12345678900014", "denomination": "Acme Corp"},
        {"siret": "98765432100028", "denomination": "Beta SAS"},
    ]

    assert _check_siret_in_list(candidates, "12345678900014")
    assert _check_siret_in_list(candidates, "98765432100028")
    assert not _check_siret_in_list(candidates, "00000000000000")
    assert not _check_siret_in_list(candidates, None)
    assert not _check_siret_in_list(candidates, "")


def test_inject_gt_if_missing():
    """Verify inject_gt_if_missing correctly appends missing GT."""
    result = CandidatePoolResult(
        candidates=[{"siret": "11111111100011", "denomination": "Existing Corp"}]
    )
    gt_cand = {"siret": "22222222200022", "denomination": "GT Corp"}

    updated = inject_gt_if_missing(result, gt_cand)
    assert len(updated.candidates) == 2
    assert updated.gt_was_injected is True

    # Calling again should not re-inject
    updated_again = inject_gt_if_missing(updated, gt_cand)
    assert len(updated_again.candidates) == 2
