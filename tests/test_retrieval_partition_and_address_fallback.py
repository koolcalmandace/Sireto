import pytest
from pathlib import Path
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.blocking import STREET_STOPWORDS, address_hash
from src.xgb_matcher.features import preprocess_crm_row, extract_street_name_from_address


def test_street_stopwords_and_address_hash() -> None:
    """Verify that extended street stopwords allow proper address hash normalization."""
    for st_type in ("COURS", "CRS", "PASSAGE", "PAS", "PROMENADE", "PROM", "VOIE", "RESIDENCE", "RES"):
        assert st_type in STREET_STOPWORDS, f"{st_type} must be in STREET_STOPWORDS"
    
    # Address hash should match between CRM with street type and SIRENE without
    h_crm = address_hash("14", "COURS DU DANUBE")
    h_sirene = address_hash("14", "DU DANUBE")
    assert h_crm == "14|DU DANUBE"
    assert h_sirene == "14|DU DANUBE"
    assert h_crm == h_sirene


def test_preprocess_crm_row_has_street_name() -> None:
    """Verify that preprocess_crm_row extracts crm_street_name correctly."""
    row = {
        "crm_name": "BOUTIQUE TEST",
        "crm_address": "14 COURS DU DANUBE",
        "crm_city": "SERRIS",
    }
    p = preprocess_crm_row(row)
    assert p["crm_street_num"] == "14"
    assert p["crm_street_name"] == "COURS DU DANUBE"


def test_partition_store_fallback_behavior(tmp_path: Path) -> None:
    """Verify that load_by_insee_then_postcode falls back to CP when INSEE partition is <= 2 rows."""
    store = PartitionedCandidateStore(tmp_path)
    
    # Mocking: load_by_insee returns 1 dummy row, load_by_postcode_filtered_insee returns 10 rows
    store.load_by_insee = lambda insee: [{"siret": "11111111111111", "insee": insee}]
    store.load_by_postcode_filtered_insee = lambda cp, insee: [{"siret": f"222222222222{i:02d}", "insee": insee} for i in range(10)]
    
    result = store.load_by_insee_then_postcode("10323", "10100")
    assert len(result) == 10
    assert result[0]["siret"] == "22222222222200"
