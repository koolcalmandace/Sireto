"""Compatibility layer for places fallback inside pipe_v6 pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .places_fallback import fallback_with_places

LOGGER = logging.getLogger(__name__)


@dataclass
class CrmRow:
    """Minimal CRM row representation for Places processing."""

    crm_id: str
    crm_name: str
    crm_address: str | None
    street_number: str | None
    street_name: str | None
    postcode: str | None
    city: str | None
    insee_code: str | None = None


@dataclass
class XgbTopkCandidate:
    """XGB top-k candidate with score."""

    siret: str
    score: float
    rank: int


@dataclass
class PlacesProcessingResult:
    """Complete result of Places-guided processing."""

    crm_id: str
    xgb_status: str
    places_response: Any
    places_decision: Any
    final_status: str  # "MATCH_PLACES", "NO_MATCH", etc.
    final_siret: str | None
    observability: Dict[str, Any]


def process_review_case(
    crm_row: CrmRow,
    xgb_topk: List[XgbTopkCandidate],
    config: Any,
    conn: Any,
    xgb_status: str,
    logger: logging.Logger | None = None,
) -> PlacesProcessingResult:
    """Adapter process_review_case using fallback_with_places."""
    log = logger or LOGGER

    # Try to load/initialize the inference engine dynamically
    try:
        from xgb_matcher.infer import XgbInferenceEngine
    except ImportError as exc:
        log.error("Could not import XgbInferenceEngine: %s", exc)
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=None,
            places_decision=None,
            final_status="NO_MATCH",
            final_siret=None,
            observability={"reason": f"import_error: {exc}", "pool_size": 0},
        )

    # Find models in models/
    ranker_path = None
    decider_path = None
    model_dir = Path("models")
    if model_dir.exists():
        ranker_files = sorted(model_dir.glob("xgbranker_*.json"), reverse=True)
        decider_files = sorted(model_dir.glob("xgb_decider_*.json"), reverse=True)
        if ranker_files:
            ranker_path = ranker_files[0]
        if decider_files:
            decider_path = decider_files[0]

    if not ranker_path or not decider_path:
        log.warning("Could not find ranker/decider models in models/ for Places fallback rerun.")
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=None,
            places_decision=None,
            final_status="NO_MATCH",
            final_siret=None,
            observability={"reason": "missing_models", "pool_size": 0},
        )

    try:
        xgb_engine = XgbInferenceEngine.from_models(
            ranker_path=ranker_path,
            decider_path=decider_path,
            partitions_dir=getattr(config, "partitions_dir", "data/candidates_v7_all"),
            risk_model_path=Path("models/routing_risk_model.pkl"),
            risk_meta_path=Path("models/routing_risk_meta.json"),
        )
    except Exception as exc:
        log.error("Failed to initialize Places fallback engine: %s", exc)
        return PlacesProcessingResult(
            crm_id=crm_row.crm_id,
            xgb_status=xgb_status,
            places_response=None,
            places_decision=None,
            final_status="NO_MATCH",
            final_siret=None,
            observability={"reason": f"engine_init_failed: {exc}", "pool_size": 0},
        )

    # Call the simplified places fallback logic
    fallback_res = fallback_with_places(
        crm_id=crm_row.crm_id,
        crm_name=crm_row.crm_name,
        crm_address=crm_row.crm_address,
        crm_postcode=crm_row.postcode,
        crm_city=crm_row.city,
        crm_insee=crm_row.insee_code,
        config=config,
        engine=xgb_engine,
        logger=log,
        cache_only=False,
    )

    # Map PlacesFallbackResult to PlacesProcessingResult structure expected by pipeline.py
    obs = {
        "reason": fallback_res.reason or "fallback_success",
        "places_used": fallback_res.places_used,
        "places_title": fallback_res.places_title,
        "places_address": fallback_res.places_address,
        "places_postcode": fallback_res.places_postcode,
        "dept_guard_passed": fallback_res.dept_guard_passed,
        "xgb_rerun_status": fallback_res.xgb_rerun_status,
        "xgb_rerun_score": fallback_res.xgb_rerun_score,
        "score_places_top1": fallback_res.xgb_rerun_score or (xgb_topk[0].score if xgb_topk else 0.0),
        "pool_size": 1 if fallback_res.places_used else 0,
    }

    return PlacesProcessingResult(
        crm_id=crm_row.crm_id,
        xgb_status=xgb_status,
        places_response=None,
        places_decision=None,
        final_status=fallback_res.final_status,
        final_siret=fallback_res.final_siret,
        observability=obs,
    )
