"""
tests/test_spatial_resolver.py
==============================
Unit tests for BAN spatial resolver and continuous distance computation.
"""

import time
import pytest
import numpy as np
from pathlib import Path

from src.xgb_matcher.spatial import SpatialResolver, DEFAULT_BAN_DB_PATH


@pytest.fixture(scope="module")
def resolver():
    res = SpatialResolver()
    if not res.is_available:
        pytest.skip(f"BAN database {DEFAULT_BAN_DB_PATH} not found.")
    return res


def test_resolver_availability(resolver):
    assert resolver.is_available is True


def test_commune_loading(resolver):
    # Test Paris 2e (75102)
    data = resolver.get_commune_data("75102")
    assert data is not None
    assert data.insee == "75102"
    assert len(data.addr_coords) > 1000
    assert len(data.street_centroids) > 50
    assert data.commune_centroid is not None
    # Latitude in Paris should be around 48.8x
    assert 48.8 <= data.commune_centroid[0] <= 48.9
    # Longitude around 2.3x
    assert 2.3 <= data.commune_centroid[1] <= 2.4


def test_geocoding_exact_address(resolver):
    # 10 Rue de la Paix in Paris 2e
    coord = resolver.geocode(
        insee="75102",
        address="10 RUE DE LA PAIX",
    )
    assert coord is not None
    lat, lon = coord
    assert 48.868 <= lat <= 48.871
    assert 2.330 <= lon <= 2.333


def test_geocoding_street_centroid(resolver):
    # Missing street number -> should resolve to street centroid
    coord = resolver.geocode(
        insee="75102",
        address="RUE DU QUATRE SEPTEMBRE",
    )
    assert coord is not None
    lat, lon = coord
    assert 48.868 <= lat <= 48.872
    assert 2.333 <= lon <= 2.340


def test_vectorized_spatial_features(resolver):
    crm_row = {
        "crm_name": "HOTEL PARK HYATT",
        "crm_address": "5 RUE DE LA PAIX",
        "insee": "75102",
        "postcode": "75002",
    }

    # 3 candidates:
    # 1. 5 Rue de la Paix (distance ~0m)
    # 2. 25 Rue de la Paix (distance ~150m, same block)
    # 3. 100 Boulevard Voltaire (far away in another district/unknown)
    candidates = [
        {
            "siret": "11100000000001",
            "numeroVoie": "5",
            "typeVoie": "RUE",
            "libelleVoie": "DE LA PAIX",
            "insee": "75102",
        },
        {
            "siret": "11100000000002",
            "numeroVoie": "25",
            "typeVoie": "RUE",
            "libelleVoie": "DE LA PAIX",
            "insee": "75102",
        },
        {
            "siret": "11100000000003",
            "numeroVoie": "50",
            "typeVoie": "RUE",
            "libelleVoie": "DU MAIL",
            "insee": "75102",
        },
    ]

    t0 = time.perf_counter()
    feat_matrix = resolver.compute_spatial_features(crm_row, candidates)
    t1 = time.perf_counter()

    assert feat_matrix.shape == (3, 4)
    # Candidate 0: distance < 0.05 km (50m)
    assert feat_matrix[0, 0] < 0.05
    assert feat_matrix[0, 1] == 1.0  # very close (<100m)
    assert feat_matrix[0, 2] == 1.0  # same block (<300m)
    assert feat_matrix[0, 3] == 1.0  # has coords

    # Candidate 1: distance between 50m and 300m
    assert 0.05 <= feat_matrix[1, 0] <= 0.30
    assert feat_matrix[1, 2] == 1.0  # same block (<300m)

    # Candidate 2: further away on another street
    assert feat_matrix[2, 0] > feat_matrix[0, 0]


def test_spatial_performance_500_candidates(resolver):
    """Verify that geocoding and computing distance on 500 candidates executes in < 30ms."""
    crm_row = {
        "crm_name": "CREDIT AGRICOLE BASTILLE",
        "crm_address": "15 RUE DE LA PAIX",
        "insee": "75102",
        "postcode": "75002",
    }
    candidates = [
        {
            "siret": f"999000{i:03d}00010",
            "numeroVoie": str((i % 30) + 1),
            "typeVoie": "RUE",
            "libelleVoie": "DE LA PAIX" if i % 2 == 0 else "DU QUATRE SEPTEMBRE",
            "insee": "75102",
        }
        for i in range(500)
    ]

    # Warm up cache
    resolver.compute_spatial_features(crm_row, candidates[:10])

    t0 = time.perf_counter()
    feat_matrix = resolver.compute_spatial_features(crm_row, candidates)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert feat_matrix.shape == (500, 4)
    assert elapsed_ms < 50.0, f"Spatial computation took {elapsed_ms:.1f}ms (expected < 50ms)"
