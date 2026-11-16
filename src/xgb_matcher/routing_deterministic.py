"""
src/xgb_matcher/routing_deterministic.py
========================================
Deterministic Natural Selection Router (Pipe V7 Stage 3).

A 100% deterministic, zero-hallucination routing policy replacing the legacy
risk metamodel. Built on 3 Cartesian pillars:

  Pillar 1: Absolute Physical Safety Floors
    - score_top1 >= 0.55 (confiance intrinsèque minimale)
    - score_gap >= 0.08 (unicité locale, anti-homonymie)
    - name_jaro >= 0.65 (ancre textuelle nominale minimale)
    - is_active == 1 (établissement en activité uniquement, etatAdministratif == 'A')

  Pillar 2: Dual-Tier Natural Selection (cible: 85-90% AUTO @ SIREN >= 99.84%)
    - Tier A (Haute Certitude Intrinsèque) :
        score_top1 >= 0.70 AND score_gap >= 0.10 AND name_jaro >= 0.75 -> AUTO
    - Tier B (Domination Contextuelle / Rival inexistant) :
        score_top1 >= 0.55 AND score_gap >= 0.15 AND
        postcode_match == 1.0 AND street_name_jaro >= 0.70 -> AUTO

  Pillar 3: Residual Ambiguity -> REVIEW
    - Tout cas ne validant ni le Palier A ni le Palier B -> REVIEW
      (sécurisé pour le Fallback Serper Places V7 sans aucun faux positif permis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class DeterministicRoutingConfig:
    """Configuration thresholds for Deterministic Natural Selection Routing."""

    # Pillar 1 : Planchers Physiques Absolus
    floor_min_score: float = 0.55
    floor_min_gap: float = 0.08
    floor_min_name_jaro: float = 0.65
    floor_require_active: bool = True

    # Pillar 2 : Palier A (Haute Certitude Intrinsèque)
    # Cible: 85-90% AUTO @ SIREN precision >= 99.84%
    tier_a_min_score: float = 0.70
    tier_a_min_gap: float = 0.10
    tier_a_min_name_jaro: float = 0.75

    # Pillar 2 : Palier B (Domination Contextuelle)
    tier_b_min_score: float = 0.55
    tier_b_min_gap: float = 0.15
    tier_b_min_street_jaro: float = 0.70
    tier_b_require_postcode: bool = True


@dataclass(frozen=True)
class DeterministicRoutingResult:
    """Structured decision output from the Deterministic Router."""

    decision: str              # "AUTO" or "REVIEW"
    tier: Optional[str]        # "TIER_A", "TIER_B", or None
    passed_floors: bool        # True if Pillar 1 floors passed
    reason: str                # Human-readable diagnostic rationale
    score_top1: float
    score_top2: float
    score_gap: float
    name_jaro: float
    street_name_jaro: float
    postcode_match: float
    is_active: bool


def route_candidate_deterministic(
    score_top1: float,
    score_top2: float,
    score_gap: Optional[float] = None,
    name_jaro: float = 0.0,
    street_name_jaro: float = 0.0,
    postcode_match: float = 0.0,
    is_active: bool = True,
    config: Optional[DeterministicRoutingConfig] = None,
) -> DeterministicRoutingResult:
    """Evaluate a candidate scene using the 3-Pillar Deterministic Routing Policy.

    Args:
        score_top1: Calibrated Stage 2 Decider score for Rank 1 candidate.
        score_top2: Calibrated Stage 2 Decider score for Rank 2 candidate (0.0 if solitary).
        score_gap: Optional explicit score gap (score_top1 - score_top2).
        name_jaro: Lexical similarity between CRM name and candidate name (0.0 to 1.0).
        street_name_jaro: Jaro similarity between CRM street and candidate street.
        postcode_match: 1.0 if postal codes match exactly, 0.0 otherwise.
        is_active: True if candidate establishment is open ('A'), False if closed ('F').
        config: Optional configuration overrides (defaults to production standards).

    Returns:
        DeterministicRoutingResult with decision ("AUTO" or "REVIEW"), tier, and diagnostics.
    """
    cfg = config or DeterministicRoutingConfig()

    s1 = float(score_top1)
    s2 = float(score_top2)
    gap = float(score_gap) if score_gap is not None else (s1 - s2)
    nj = float(name_jaro)
    sj = float(street_name_jaro)
    pm = float(postcode_match)
    act = bool(is_active)

    # -------------------------------------------------------------------------
    # Pillar 1 : Planchers Physiques Absolus (Garde-fous Zéro Hallucination)
    # -------------------------------------------------------------------------
    failed_reasons: List[str] = []

    if cfg.floor_require_active and not act:
        failed_reasons.append("ESTABLISHMENT_CLOSED")
    if s1 < cfg.floor_min_score:
        failed_reasons.append(f"SCORE_BELOW_FLOOR({s1:.4f} < {cfg.floor_min_score})")
    if gap < cfg.floor_min_gap:
        failed_reasons.append(f"GAP_BELOW_FLOOR({gap:.4f} < {cfg.floor_min_gap})")
    if nj < cfg.floor_min_name_jaro:
        failed_reasons.append(f"NAME_JARO_BELOW_FLOOR({nj:.4f} < {cfg.floor_min_name_jaro})")

    if failed_reasons:
        return DeterministicRoutingResult(
            decision="REVIEW",
            tier=None,
            passed_floors=False,
            reason="FLOOR_REJECT: " + "; ".join(failed_reasons),
            score_top1=s1,
            score_top2=s2,
            score_gap=gap,
            name_jaro=nj,
            street_name_jaro=sj,
            postcode_match=pm,
            is_active=act,
        )

    # -------------------------------------------------------------------------
    # Pillar 2 : Double Palier de Sélection Naturelle
    # -------------------------------------------------------------------------

    # Palier A : Haute Certitude Intrinsèque
    if (
        s1 >= cfg.tier_a_min_score
        and gap >= cfg.tier_a_min_gap
        and nj >= cfg.tier_a_min_name_jaro
    ):
        return DeterministicRoutingResult(
            decision="AUTO",
            tier="TIER_A",
            passed_floors=True,
            reason=f"AUTO_TIER_A: High intrinsic confidence (score={s1:.3f}, gap={gap:.3f}, name_jaro={nj:.3f})",
            score_top1=s1,
            score_top2=s2,
            score_gap=gap,
            name_jaro=nj,
            street_name_jaro=sj,
            postcode_match=pm,
            is_active=act,
        )

    # Palier B : Domination Contextuelle (score modéré, rival inexistant, adresse physique ferme)
    postcode_ok = (pm >= 1.0) if cfg.tier_b_require_postcode else True
    if (
        s1 >= cfg.tier_b_min_score
        and gap >= cfg.tier_b_min_gap
        and postcode_ok
        and sj >= cfg.tier_b_min_street_jaro
    ):
        return DeterministicRoutingResult(
            decision="AUTO",
            tier="TIER_B",
            passed_floors=True,
            reason=f"AUTO_TIER_B: Contextual domination (score={s1:.3f}, gap={gap:.3f}, street_jaro={sj:.3f})",
            score_top1=s1,
            score_top2=s2,
            score_gap=gap,
            name_jaro=nj,
            street_name_jaro=sj,
            postcode_match=pm,
            is_active=act,
        )

    # -------------------------------------------------------------------------
    # Pillar 3 : Zone de Doute Résiduelle -> REVIEW
    # -------------------------------------------------------------------------
    return DeterministicRoutingResult(
        decision="REVIEW",
        tier=None,
        passed_floors=True,
        reason=f"REVIEW_AMBIGUOUS: Valid floors but unmet tier criteria (score={s1:.3f}, gap={gap:.3f}, name_jaro={nj:.3f}, street_jaro={sj:.3f})",
        score_top1=s1,
        score_top2=s2,
        score_gap=gap,
        name_jaro=nj,
        street_name_jaro=sj,
        postcode_match=pm,
        is_active=act,
    )


def route_from_feature_vector(
    top1_features: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    score_top1: float,
    score_top2: float,
    candidate_dict: Optional[Mapping[str, Any]] = None,
    config: Optional[DeterministicRoutingConfig] = None,
) -> DeterministicRoutingResult:
    """Route using a precomputed Stage 2 feature vector and candidate dictionary.

    Args:
        top1_features: Array of 67 Stage 2 features for the Top-1 candidate.
        feature_names: Sequence of feature names corresponding to top1_features.
        score_top1: Calibrated Stage 2 Decider score for Rank 1.
        score_top2: Calibrated Stage 2 Decider score for Rank 2.
        candidate_dict: Raw candidate dictionary (for etat_admin fallback).
        config: Optional configuration overrides.

    Returns:
        DeterministicRoutingResult.
    """
    feat_map: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if i < len(top1_features):
            feat_map[name] = float(top1_features[i])

    # Name similarity extraction (prioritizes name_jaro_max or name_sim_max_etab)
    name_jaro = feat_map.get("name_jaro_max", 0.0)
    if name_jaro == 0.0:
        name_jaro = feat_map.get("name_sim_max_etab", 0.0)

    # Street and postcode features
    street_name_jaro = feat_map.get("street_name_jaro", 0.0)
    postcode_match = feat_map.get("postcode_match", 0.0)

    # Establishment active status
    is_active_feat = feat_map.get("is_active", 1.0)
    if candidate_dict is not None:
        etat = str(candidate_dict.get("etat_admin") or candidate_dict.get("etatAdministratifEtablissement") or "").strip().upper()
        if etat:
            is_active = (etat == "A")
        else:
            is_active = bool(is_active_feat > 0.5)
    else:
        is_active = bool(is_active_feat > 0.5)

    return route_candidate_deterministic(
        score_top1=score_top1,
        score_top2=score_top2,
        name_jaro=name_jaro,
        street_name_jaro=street_name_jaro,
        postcode_match=postcode_match,
        is_active=is_active,
        config=config,
    )

