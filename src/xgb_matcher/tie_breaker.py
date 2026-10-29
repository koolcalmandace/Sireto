"""
Stage 2 Tie-Breaker Engine for SIRETO Pipe V7.

Deterministic reranking for Top-5 candidates to resolve near-misses:
  1. Intra-SIREN Conflict: Siège vs Secondary Establishment matching the physical address.
  2. Inter-SIREN Specificity: High locality suffix or exact spatial anchor when ML scores are tied.
  3. Safety Guards: Minimum score floor, active status enforcement, zero regression tolerance.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .features import FEATURE_NAMES

LOGGER = logging.getLogger(__name__)

# Precomputed feature name to index map
_FEAT_MAP = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _get_val(feat_row: np.ndarray, feat_name: str, default: float = 0.0) -> float:
    idx = _FEAT_MAP.get(feat_name)
    if idx is None or idx >= len(feat_row):
        return default
    val = feat_row[idx]
    if np.isnan(val):
        return default
    return float(val)


def _safe_siret(cand: Dict[str, Any]) -> str:
    raw = cand.get("siret") or cand.get("siret_etab") or ""
    return str(raw).strip().zfill(14)


def _safe_siren(cand: Dict[str, Any]) -> str:
    siret = _safe_siret(cand)
    if len(siret) >= 9:
        return siret[:9]
    raw = cand.get("siren") or ""
    return str(raw).strip().zfill(9)


def apply_stage2_tie_breakers(
    retained_cands: List[Dict[str, Any]],
    s2_scores: np.ndarray,
    s2_matrix: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_k: int = 5,
    min_score_floor: float = 0.25,
    intra_siren_max_gap: float = 0.20,
    inter_siren_max_gap: float = 0.06,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Rerank Stage 2 candidates deterministically to resolve near-misses.

    Parameters:
        retained_cands: List of candidate dictionaries.
        s2_scores: 1D array of calibrated (or raw) Stage 2 scores.
        s2_matrix: 2D array of shape (N, num_features) corresponding to FEATURE_NAMES.
        feature_names: Optional feature name order (defaults to FEATURE_NAMES).
        max_k: Maximum candidate rank to evaluate for promotion (default: 5).
        min_score_floor: Minimum score required for a candidate to be eligible for promotion.
        intra_siren_max_gap: Maximum score difference for Intra-SIREN tie-breaking (default: 0.20).
        inter_siren_max_gap: Maximum score difference for Inter-SIREN tie-breaking (default: 0.06).

    Returns:
        (new_order_indices, meta_dict)
    """
    if not retained_cands or len(retained_cands) <= 1:
        order = list(range(len(retained_cands)))
        return order, {"applied": False, "reason": "single_candidate_or_empty"}

    # Base ordering by Stage 2 scores (descending)
    initial_order = list(np.argsort(-s2_scores))
    top1_idx = initial_order[0]
    top1_score = float(s2_scores[top1_idx])
    top1_cand = retained_cands[top1_idx]
    top1_siret = _safe_siret(top1_cand)
    top1_siren = _safe_siren(top1_cand)
    top1_feat = s2_matrix[top1_idx]

    # Pre-extract top1 indicators
    top1_is_siege = (_get_val(top1_feat, "is_siege") >= 0.5 or bool(top1_cand.get("is_siege")))
    top1_exact_anchor = (_get_val(top1_feat, "exact_address_anchor") >= 0.5)
    top1_num_match = (_get_val(top1_feat, "street_number_match") >= 0.5)
    top1_city_suffix = _get_val(top1_feat, "name_city_suffix_match")
    top1_addr_jaro = _get_val(top1_feat, "addr_jaro")

    candidates_to_check = initial_order[1:max_k]
    promoted_idx: Optional[int] = None
    applied_rule: Optional[str] = None

    # -------------------------------------------------------------------------
    # Rule 1: Intra-SIREN Conflict (Siège vs Local Secondary Establishment)
    # -------------------------------------------------------------------------
    intra_candidates = []
    for cand_idx in candidates_to_check:
        cand = retained_cands[cand_idx]
        cand_siren = _safe_siren(cand)
        cand_score = float(s2_scores[cand_idx])
        cand_feat = s2_matrix[cand_idx]

        # Valid 9-digit SIREN check
        if not cand_siren or not top1_siren or cand_siren == "000000000":
            continue
        if cand_siren != top1_siren:
            continue

        # Score floor & max gap guard
        if cand_score < min_score_floor:
            continue
        if (top1_score - cand_score) > intra_siren_max_gap:
            continue

        # Must be active
        cand_is_active = (_get_val(cand_feat, "is_active") >= 0.5 or cand.get("etatAdministratifEtablissement") == "A")
        if not cand_is_active:
            continue

        cand_exact_anchor = (_get_val(cand_feat, "exact_address_anchor") >= 0.5)
        cand_num_match = (_get_val(cand_feat, "street_number_match") >= 0.5)
        cand_street_jaro = _get_val(cand_feat, "street_name_jaro")
        cand_delta_siege = _get_val(cand_feat, "delta_addr_vs_siege")
        cand_is_siege = (_get_val(cand_feat, "is_siege") >= 0.5 or bool(cand.get("is_siege")))

        # Candidate must NOT be another siege
        if cand_is_siege:
            continue

        # Condition for Intra-SIREN promotion:
        # 1. Top 1 is the Siège (or candidate has a decisive positive delta vs siege),
        # 2. Candidate has exact street number and high street name similarity,
        # 3. Top 1 does NOT have an exact street number match or exact anchor.
        cand_has_strong_addr = cand_exact_anchor or (cand_num_match and cand_street_jaro >= 0.85)
        top1_lacks_addr = (not top1_num_match) and (not top1_exact_anchor)

        if cand_has_strong_addr and top1_lacks_addr and (top1_is_siege or cand_delta_siege >= 0.15):
            priority = (10.0 if cand_exact_anchor else 5.0) + cand_street_jaro + cand_score
            intra_candidates.append((cand_idx, priority))

    if intra_candidates:
        intra_candidates.sort(key=lambda x: x[1], reverse=True)
        promoted_idx = intra_candidates[0][0]
        applied_rule = "INTRA_SIREN_LOCAL_ESTABLISHMENT"

    # -------------------------------------------------------------------------
    # Rule 2: Inter-SIREN Specificity Tie-Breaker (If no intra-siren promotion)
    # -------------------------------------------------------------------------
    if promoted_idx is None:
        inter_candidates = []
        for cand_idx in candidates_to_check:
            cand = retained_cands[cand_idx]
            cand_siren = _safe_siren(cand)
            cand_score = float(s2_scores[cand_idx])
            cand_feat = s2_matrix[cand_idx]

            # Only for distinct SIRENs
            if cand_siren and top1_siren and cand_siren == top1_siren:
                continue

            # Gap check: Must be a true near tie
            score_gap = top1_score - cand_score
            if score_gap > inter_siren_max_gap:
                continue

            if cand_score < min_score_floor:
                continue

            cand_is_active = (_get_val(cand_feat, "is_active") >= 0.5 or cand.get("etatAdministratifEtablissement") == "A")
            if not cand_is_active:
                continue

            cand_exact_anchor = (_get_val(cand_feat, "exact_address_anchor") >= 0.5)
            cand_city_suffix = _get_val(cand_feat, "name_city_suffix_match")
            cand_name_jaro = _get_val(cand_feat, "name_jaro_max")
            cand_addr_jaro = _get_val(cand_feat, "addr_jaro")

            is_eligible = False
            priority = 0.0

            # Sub-rule 2a: Name locality suffix match
            # e.g., CRM "OPHIS 63 (BLANZAT)" vs candidate having city suffix
            if cand_city_suffix >= 0.50 and top1_city_suffix < 0.10 and cand_addr_jaro >= 0.70:
                is_eligible = True
                priority = 8.0 + cand_city_suffix + cand_score

            # Sub-rule 2b: Exact Address Anchor vs No Anchor on Top 1
            elif cand_exact_anchor and not top1_exact_anchor and cand_name_jaro >= 0.75:
                is_eligible = True
                priority = 6.0 + cand_name_jaro + cand_score

            if is_eligible:
                inter_candidates.append((cand_idx, priority))

        if inter_candidates:
            inter_candidates.sort(key=lambda x: x[1], reverse=True)
            promoted_idx = inter_candidates[0][0]
            applied_rule = "INTER_SIREN_SPECIFICITY_MATCH"

    # -------------------------------------------------------------------------
    # Apply Reranking if Promoted
    # -------------------------------------------------------------------------
    if promoted_idx is not None and applied_rule is not None:
        new_order = [promoted_idx] + [idx for idx in initial_order if idx != promoted_idx]
        promoted_cand = retained_cands[promoted_idx]
        return new_order, {
            "applied": True,
            "rule": applied_rule,
            "promoted_idx": promoted_idx,
            "promoted_siret": _safe_siret(promoted_cand),
            "prev_top1_idx": top1_idx,
            "prev_top1_siret": top1_siret,
            "score_promoted": float(s2_scores[promoted_idx]),
            "score_prev_top1": top1_score,
        }

    return initial_order, {"applied": False, "reason": "no_tie_breaker_triggered"}
