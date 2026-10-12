"""
src/xgb_matcher/retention.py
============================
Top-K retention safeguards for Stage 1 Fast Ranker.
Ensures candidate pool quality from K=500 (Retrieval) to K=50 (Fast Ranker)
under a strict K=50 budget.

Safeguards:
  1. Siège Social & Sidecar Preservation:
     - Never evict a confirmed Sidecar match or the head office of an active SIREN.
  2. Guard 1: Signal-Gated Intra-SIREN Compaction:
     - If a SIREN has >= 3 candidates in the top-50, strictly protect any establishment
       with a specific local signal (address match, house number match, enseigne match, or siège).
     - Compact redundant "passive clones" (annexes with zero address/enseigne affinity) to max 1,
       freeing slots for third-party entities.
  3. Guard 2: Address Density Dampening:
     - On high-density addresses (business centers / domiciliation >= 5 companies),
       deprioritize passive co-tenants that have zero nominal affinity (name_jaro < 0.35).
  4. Guard 3: IDF Specificity Anchor & High-Name Rescue:
     - Promote candidates in ranks 51-200 sharing rare distinctive tokens (IDF >= 7.0, len >= 4)
       or high name similarity (>= 0.85) into freed slots.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from .features import (
    FAST_RANKER_FEATURE_NAMES,
    NAME_STOPWORDS,
    _GLOBAL_NAME_IDF,
    _GLOBAL_NAME_IDF_DEFAULT,
    _tokenize,
    jaro_sim,
    normalize_name,
    normalize_text,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionGuardConfig:
    """Configuration parameters for Fast Ranker retention safeguards."""
    top_k: int = 50
    tail_search_max: int = 200
    max_swaps: int = 8
    enable_siege_promotion: bool = False  # Disabled by default to protect secondary establishments

    # Guard 1: Signal-Gated Intra-SIREN Compaction
    intra_siren_trigger_count: int = 3
    intra_siren_min_rank_eviction: int = 15  # Candidates in top-15 raw are immune from clone eviction
    signal_street_name_jaro_min: float = 0.75
    signal_addr_jaro_min: float = 0.80
    signal_enseigne_jaro_min: float = 0.70

    # Guard 2: Address Density Dampening
    density_threshold: float = 5.0
    density_name_jaro_min: float = 0.35

    # Guard 3: IDF Rarity Anchor & High-Name Rescue
    idf_anchor_min: float = 7.0
    idf_token_min_len: int = 4
    name_rescue_jaro_min: float = 0.85


def _extract_tokens(text: str, min_len: int = 3) -> Set[str]:
    """Helper to tokenize text on alphanumeric word boundaries, removing common stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"[A-Z0-9]+", str(text).upper())
    return {
        tok for tok in tokens
        if len(tok) >= min_len and tok not in NAME_STOPWORDS and not tok.isdigit()
    }


def apply_retention_topk(
    candidates: List[Dict[str, Any]],
    scores: np.ndarray,
    top_k: int = 50,
    config: Optional[RetentionGuardConfig] = None,
    feat_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    crm_row: Optional[Mapping[str, Any]] = None,
    name_idf_map: Optional[Mapping[str, float]] = None,
) -> Tuple[List[int], List[str]]:
    """
    Ranks candidates by XGBoost score and applies 3-stage retention safeguards:
      - Preserves Sidecar Anchors and high-confidence nominal matches.
      - Guard 1: Signal-gated intra-SIREN compaction (prevents passive clone clutter in tail).
      - Guard 2: Address density dampening (evicts pure co-tenants without name link).
      - Guard 3: IDF rarity anchor (promotes rare-token & high-similarity candidates).

    Args:
        candidates: Candidate dictionaries from SIRENE retrieval.
        scores: Predicted XGBoost scores (1D array of shape [n_candidates]).
        top_k: Target number of candidates to retain (default 50).
        config: Optional RetentionGuardConfig.
        feat_matrix: Optional precomputed feature matrix [n_candidates, n_features].
        feature_names: Optional feature column names matching feat_matrix.
        crm_row: Optional CRM query row (dict or Series).
        name_idf_map: Optional IDF map for token rarity.

    Returns:
        retained_indices: List of candidate indices (length <= top_k).
        retained_sirets: List of 14-digit SIRET strings.
    """
    cfg = config or RetentionGuardConfig(top_k=top_k)
    n_cands = len(candidates)
    if n_cands <= cfg.top_k:
        order = list(np.argsort(-scores))
        return order, [str(candidates[i].get("siret") or "").strip().zfill(14) for i in order]

    order = list(np.argsort(-scores))
    selected_indices = list(order[:cfg.top_k])
    selected_sirets = [str(candidates[i].get("siret") or "").strip().zfill(14) for i in selected_indices]
    selected_sirens = [s[:9] for s in selected_sirets if len(s) >= 9]
    siren_counts = Counter(selected_sirens)

    # Fast column lookup in precomputed feature matrix if available
    f_cols: Dict[str, int] = {}
    if feat_matrix is not None and feature_names is not None:
        f_cols = {fn: idx for idx, fn in enumerate(feature_names)}

    def _get_val(cand_idx: int, col_name: str, fallback_key: str = "", default: float = 0.0) -> float:
        if feat_matrix is not None and col_name in f_cols:
            return float(feat_matrix[cand_idx, f_cols[col_name]])
        cand = candidates[cand_idx]
        val = cand.get(col_name)
        if val is None and fallback_key:
            val = cand.get(fallback_key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return default

    # Extract CRM tokens and address presence
    crm_name = ""
    crm_tokens: Set[str] = set()
    crm_has_street = False
    if crm_row is not None:
        crm_name = str(crm_row.get("crm_name") or crm_row.get("Client final") or "")
        crm_tokens = _extract_tokens(crm_name, min_len=cfg.idf_token_min_len)
        addr_str = str(crm_row.get("crm_adresse") or crm_row.get("crm_address") or crm_row.get("Adresse") or "").strip()
        st_str = str(crm_row.get("crm_street_name") or "").strip()
        crm_has_street = (len(addr_str) >= 3 and addr_str.lower() != "nan") or (len(st_str) >= 3 and st_str.lower() != "nan")

    effective_idf_map = name_idf_map if name_idf_map is not None else _GLOBAL_NAME_IDF

    # -------------------------------------------------------------------------
    # 1. Identify Candidate Protections in Tail (ranks top_k to min(n_cands, tail_search_max))
    # -------------------------------------------------------------------------
    tail_candidates_to_promote: List[Tuple[int, str]] = []
    seen_promoted: Set[int] = set()

    for rank_idx in range(cfg.top_k, min(n_cands, cfg.tail_search_max)):
        cand_idx = order[rank_idx]
        c = candidates[cand_idx]
        s = str(c.get("siret") or "").strip().zfill(14)
        siren = s[:9]

        # Rule A: Optional Siège Social promotion (if enabled)
        if cfg.enable_siege_promotion:
            is_siege = bool(c.get("is_siege", False)) or (_get_val(cand_idx, "is_siege", "is_siege") == 1.0)
            if is_siege and siren in siren_counts:
                tail_candidates_to_promote.append((10.0 + float(scores[cand_idx]), cand_idx, "SIEGE_RETENTION"))
                seen_promoted.add(cand_idx)
                continue

        is_sidecar = bool(c.get("_is_sidecar_match") or c.get("is_sidecar_match"))

        # Rule B: Validated Sidecar match (highest priority)
        if is_sidecar:
            tail_candidates_to_promote.append((100.0 + float(scores[cand_idx]), cand_idx, "SIDECAR_RETENTION"))
            seen_promoted.add(cand_idx)
            continue

        # Rule C: Guard 3 - IDF Specificity Anchor (rare distinct token shared with CRM)
        cand_name_fields = [
            c.get("denomination"),
            c.get("denomination_ul"),
            c.get("denominationUniteLegale"),
            c.get("enseigne1"),
            c.get("enseigne1Etablissement"),
            c.get("denominationUsuelleEtablissement"),
            c.get("nom_ul"),
            c.get("nomUniteLegale"),
            c.get("prenom_usuel_ul"),
            c.get("sigle"),
            c.get("sigleUniteLegale"),
        ]
        cand_text = " ".join(str(f) for f in cand_name_fields if f)
        cand_tokens = _extract_tokens(cand_text, min_len=cfg.idf_token_min_len)

        overlap_tokens = crm_tokens & cand_tokens
        has_rare_token = False
        max_idf = 0.0
        if overlap_tokens and effective_idf_map:
            max_idf = max(effective_idf_map.get(tok, _GLOBAL_NAME_IDF_DEFAULT) for tok in overlap_tokens)
            if max_idf >= cfg.idf_anchor_min:
                has_rare_token = True

        name_jaro = _get_val(cand_idx, "name_jaro_max", "name_jaro_max", 0.0)
        if not name_jaro and crm_name and cand_text:
            name_jaro = jaro_sim(crm_name, normalize_name(cand_text))

        if has_rare_token:
            tail_candidates_to_promote.append((50.0 + max_idf, cand_idx, "IDF_ANCHOR_RETENTION"))
            seen_promoted.add(cand_idx)
            continue

        # Rule D: High-Confidence Name Match pushed into tail
        if name_jaro >= cfg.name_rescue_jaro_min:
            postcode_m = _get_val(cand_idx, "postcode_match", "postcode_match", 1.0)
            if postcode_m >= 1.0:
                tail_candidates_to_promote.append((30.0 + name_jaro * 10.0, cand_idx, "HIGH_NAME_RESCUE"))
                seen_promoted.add(cand_idx)
                continue

    if not tail_candidates_to_promote:
        return selected_indices, selected_sirets

    # Sort candidates to promote by priority descending (strongest signal first)
    tail_candidates_to_promote.sort(key=lambda x: x[0], reverse=True)

    # -------------------------------------------------------------------------
    # 2. Identify Eviction Candidates within Current Top-K (Guards 1 & 2)
    # -------------------------------------------------------------------------
    # Multi-site SIREN tracking for Guard 1
    siren_members_in_topk: Dict[str, List[int]] = {}
    for slot_idx, cand_idx in enumerate(selected_indices):
        s = selected_sirets[slot_idx]
        siren = s[:9]
        if siren_counts[siren] >= cfg.intra_siren_trigger_count:
            siren_members_in_topk.setdefault(siren, []).append(slot_idx)

    eviction_priority_slots: List[int] = []

    # 2A. Guard 1: Redundant passive clones of multi-site SIRENs (tail of top-k only)
    for siren, slot_list in siren_members_in_topk.items():
        passive_clone_slots: List[int] = []
        for slot in slot_list:
            if slot < cfg.intra_siren_min_rank_eviction:
                # Top-ranked candidates are immune from passive clone eviction
                continue
            c_idx = selected_indices[slot]
            cand = candidates[c_idx]

            is_siege = bool(cand.get("is_siege", False)) or (_get_val(c_idx, "is_siege", "is_siege") == 1.0)
            is_sidecar = bool(cand.get("_is_sidecar_match") or cand.get("is_sidecar_match"))
            street_jaro = _get_val(c_idx, "street_name_jaro", "street_name_jaro", 0.0)
            street_num_m = _get_val(c_idx, "street_number_match", "street_number_match", 0.0)
            addr_jaro = _get_val(c_idx, "addr_jaro", "addr_jaro", 0.0)
            enseigne_jaro = _get_val(c_idx, "name_jaro_vs_enseigne", "name_jaro_vs_enseigne", 0.0)

            # Signal Check: Siège, Sidecar, Street, House Number, Address, or Enseigne
            has_positive_signal = (
                is_siege
                or is_sidecar
                or (crm_has_street and street_jaro >= cfg.signal_street_name_jaro_min)
                or (crm_has_street and street_num_m >= 1.0)
                or (crm_has_street and addr_jaro >= cfg.signal_addr_jaro_min)
                or (enseigne_jaro >= cfg.signal_enseigne_jaro_min)
            )

            # If CRM has no street, do not penalize candidates with strong name similarity
            if not crm_has_street:
                name_jaro = _get_val(c_idx, "name_jaro_max", "name_jaro_max", 0.0)
                if name_jaro >= 0.70 or enseigne_jaro >= cfg.signal_enseigne_jaro_min:
                    has_positive_signal = True

            if not has_positive_signal:
                passive_clone_slots.append(slot)

        # Preserve the top passive clone, mark subsequent clones as eviction slots
        if len(passive_clone_slots) > 1:
            for evict_slot in passive_clone_slots[1:]:
                if evict_slot not in eviction_priority_slots:
                    eviction_priority_slots.append(evict_slot)

    # 2B. Guard 2: Passive Co-tenants at high-density addresses
    for slot in range(cfg.top_k - 1, -1, -1):
        if slot in eviction_priority_slots:
            continue
        c_idx = selected_indices[slot]
        cand = candidates[c_idx]

        is_sidecar = bool(cand.get("_is_sidecar_match") or cand.get("is_sidecar_match"))
        if is_sidecar:
            continue

        density = _get_val(c_idx, "address_density", "_xgb_addr_density_insee", 1.0)
        name_jaro = _get_val(c_idx, "name_jaro_max", "name_jaro_max", 0.0)
        tok_overlap = _get_val(c_idx, "name_token_overlap_max", "name_token_overlap_max", 0.0)

        # High address density with near-zero name correlation
        if density >= cfg.density_threshold and name_jaro < cfg.density_name_jaro_min and tok_overlap == 0.0:
            eviction_priority_slots.append(slot)

    # 2C. Fallback: Bottom slots of top-k without sidecar or strong name protection
    for slot in range(cfg.top_k - 1, max(cfg.intra_siren_min_rank_eviction, 0), -1):
        if slot in eviction_priority_slots:
            continue
        c_idx = selected_indices[slot]
        cand = candidates[c_idx]
        is_sidecar = bool(cand.get("_is_sidecar_match") or cand.get("is_sidecar_match"))
        name_jaro = _get_val(c_idx, "name_jaro_max", "name_jaro_max", 0.0)
        tok_overlap = _get_val(c_idx, "name_token_overlap_max", "name_token_overlap_max", 0.0)
        has_name_signal = (name_jaro >= 0.70) or (tok_overlap > 0.0)

        if not is_sidecar and not has_name_signal:
            eviction_priority_slots.append(slot)

    # -------------------------------------------------------------------------
    # 3. Apply Swaps (Budget strictly capped at top_k and max_swaps)
    # -------------------------------------------------------------------------
    final_indices = list(selected_indices)
    swaps_applied = 0
    for priority, cand_idx, reason in tail_candidates_to_promote:
        if swaps_applied >= cfg.max_swaps:
            break
        if not eviction_priority_slots:
            break

        slot_to_replace = eviction_priority_slots.pop(0)
        final_indices[slot_to_replace] = cand_idx
        swaps_applied += 1

    final_sirets = [str(candidates[i].get("siret") or "").strip().zfill(14) for i in final_indices]
    return final_indices, final_sirets
