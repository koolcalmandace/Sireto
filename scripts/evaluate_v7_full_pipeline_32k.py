#!/usr/bin/env python3
"""
scripts/evaluate_v7_full_pipeline_32k.py
==========================================
Comprehensive End-to-End Pipeline V7 Evaluator on National 32K Dataset.

Evaluates the complete 3-Stage Architecture:
  1. Stage 1: Fast Ranker V9 (NDCG@50, 47 features, Dual-Preservation, Upper-Tier Sanctuary)
  2. Stage 2: Unified XGB Decider (67 features, NAF vector, delta_addr_vs_siege, civil_name_jaro, Isotonic)
  3. Stage 3: Routing Risk Model (Scene metamodel, AUTO @ 99.84% vs REVIEW)

Features:
  - Atomic Checkpointing & Auto-Resume (--resume)
  - MemoryGovernor (RAM < 3.5 GB)
  - PowerMonitor (Windows sleep / lid-close / lock event detection & anti-sleep)
  - Geographic clustering in RAM for Parquet cache locality (>99% hit rate)
"""

from __future__ import annotations

import os
os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")

import argparse
import csv
import gc
import json
import logging
import pickle
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.calibration import IsotonicCalibrator, SigmoidCalibrator, load_calibrator
from src.xgb_matcher.checkpoint import CheckpointManager

sys.modules["__main__"].IsotonicCalibrator = IsotonicCalibrator
sys.modules["__main__"].SigmoidCalibrator = SigmoidCalibrator
from src.xgb_matcher.features import (
    FAST_RANKER_FEATURE_NAMES,
    FEATURE_NAMES,
    make_features_batch,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.memory_governor import MemoryGovernor
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.power_guard import PowerMonitor
from src.xgb_matcher.retention import apply_retention_topk
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.routing_deterministic import (
    DeterministicRoutingConfig,
    DeterministicRoutingResult,
    route_from_feature_vector,
)
from src.xgb_matcher.routing_risk import BASE_FEATURES, build_feature_row, default_feature_columns
from src.xgb_matcher.spatial import SpatialResolver
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
from src.xgb_matcher.tie_breaker import apply_stage2_tie_breakers

logger = logging.getLogger("evaluate_v7_full_pipeline_32k")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def safe_str(val: Any) -> str:
    """Safely converts any value (including NaN/float/None) to a clean stripped string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def cluster_rows_by_locality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups rows by commune in-memory to maximize Parquet store and TF-IDF cache hits."""
    commune_buckets: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    for r in rows:
        insee = safe_str(r.get("crm_insee") or r.get("insee"))
        cp = safe_str(r.get("crm_cp") or r.get("postcode"))
        key = f"insee_{insee}" if insee else (f"cp_{cp}" if cp else "unknown")
        commune_buckets.setdefault(key, []).append(r)
    clustered: List[Dict[str, Any]] = []
    for bucket in commune_buckets.values():
        clustered.extend(bucket)
    return clustered


def run_pipeline_evaluation(
    crm_path: Path,
    partitions_dir: Path,
    output_dir: Path,
    tag: str,
    ranker_path: Path,
    decider_path: Path,
    calibrator_path: Optional[Path],
    risk_model_path: Optional[Path],
    risk_calibrator_path: Optional[Path] = None,
    risk_meta_path: Optional[Path] = None,
    ban_db_path: Path = PROJECT_ROOT / "data" / "ban.duckdb",
    prefilter_k: int = 500,
    top_k: int = 50,
    cache_size: int = 40,
    max_ram_gb: float = 3.5,
    checkpoint_interval: int = 250,
    resume: bool = True,
    max_rows: Optional[int] = None,
    log_file: Optional[Path] = None,
    no_pause_on_lock: bool = True,
    routing_mode: str = "deterministic",
    deterministic_config: Optional[DeterministicRoutingConfig] = None,
    stage2_ranker_path: Optional[Path] = None,
    use_tie_breaker: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / ".checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{tag}_summary.json"
    details_path = output_dir / f"{tag}_details.csv"

    print("=" * 80)
    print("SIRETO V7 — BENCHMARK NATIONAL COMPLET 32K (STAGE 1 + STAGE 2 + STAGE 3)")
    print(f"Tag             : {tag}")
    print(f"Routing Mode S3 : {routing_mode.upper()}")
    print(f"Stage 1 Ranker  : {ranker_path}")
    if stage2_ranker_path:
        print(f"Stage 2 Ranker  : {stage2_ranker_path}")
    print(f"Stage 2 Decider : {decider_path}")
    print(f"Calibrateur S2  : {calibrator_path}")
    print(f"Tie-Breaker S2  : {'ON' if use_tie_breaker else 'OFF (Pure ML)'}")
    if routing_mode == "risk":
        print(f"Risk Model S3   : {risk_model_path}")
        print(f"Risk Calib S3   : {risk_calibrator_path}")
    else:
        print("Deterministic   : 3-Pillar Natural Selection (Pillar 1 Floors, Tier A, Tier B)")
    print("=" * 80)

    # 1. Chargement des modèles
    print("\n[1/4] Chargement des modèles XGBoost Stage 1, Stage 2 et Stage 3...")
    ranker_booster = xgb.Booster()
    ranker_booster.load_model(str(ranker_path))

    decider_booster = xgb.Booster()
    decider_booster.load_model(str(decider_path))

    stage2_ranker_booster = None
    if stage2_ranker_path and stage2_ranker_path.exists():
        stage2_ranker_booster = xgb.Booster()
        stage2_ranker_booster.load_model(str(stage2_ranker_path))

    isotonic_calibrator = None
    if calibrator_path and calibrator_path.exists():
        isotonic_calibrator = load_calibrator(calibrator_path)

    risk_model = None
    risk_calibrator = None
    risk_threshold = 0.925
    risk_feature_names = default_feature_columns(BASE_FEATURES)
    if risk_meta_path and risk_meta_path.exists():
        with open(risk_meta_path, "r", encoding="utf-8") as f:
            rmeta = json.load(f)
            risk_threshold = rmeta.get("threshold", 0.925)
            risk_feature_names = rmeta.get("features", risk_feature_names)
            if not risk_calibrator_path and rmeta.get("calibrator_path"):
                cal_p = Path(rmeta.get("calibrator_path"))
                if cal_p.exists():
                    risk_calibrator_path = cal_p
                elif (PROJECT_ROOT / cal_p).exists():
                    risk_calibrator_path = PROJECT_ROOT / cal_p

    if risk_calibrator_path and risk_calibrator_path.exists():
        try:
            risk_calibrator = load_calibrator(risk_calibrator_path)
            logger.info("Calibrateur de risque Stage 3 chargé avec succès: %s", risk_calibrator_path)
        except Exception as e:
            logger.warning("Échec du chargement du calibrateur de risque: %s", e)

    if risk_model_path and risk_model_path.exists():
        with open(risk_model_path, "rb") as f:
            risk_model = pickle.load(f)

    # 2. Chargement du dataset CRM
    print("\n[2/4] Chargement et clustering géographique des requêtes CRM...")
    df_crm = pd.read_csv(crm_path, sep=";", dtype=str, keep_default_na=False)
    all_rows = df_crm.to_dict(orient="records")
    if max_rows and max_rows > 0:
        all_rows = all_rows[:max_rows]

    fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

    # Checkpointing et reprise
    ckpt = CheckpointManager(
        tag=tag,
        checkpoint_dir=checkpoints_dir,
        total_rows=len(all_rows),
        crm_path=crm_path,
        save_interval=checkpoint_interval,
    )
    is_resumed = False
    processed_crm_ids: Set[str] = set()

    if resume and details_path.exists() and details_path.stat().st_size > 0:
        processed_crm_ids = ckpt.get_processed_crm_ids(details_path)
        if processed_crm_ids:
            is_resumed = True
            print(f"[REPRISE ATOMIQUE] Reprise active : {len(processed_crm_ids)} requêtes déjà traitées depuis le CSV de détails.")
    elif resume and ckpt.is_resumable():
        ckpt.load()
        processed_crm_ids = ckpt.get_processed_crm_ids(details_path)
        is_resumed = True
        print(f"[REPRISE ATOMIQUE] Reprise active : {len(processed_crm_ids)} requêtes déjà traitées.")

    unprocessed = [r for r in all_rows if safe_str(r.get("crm_id") or r.get("id")) not in processed_crm_ids]
    eval_rows = cluster_rows_by_locality(unprocessed)
    print(f"  Total requêtes à traiter : {len(eval_rows)} / {len(all_rows)} ({len(processed_crm_ids)} déjà complétées)")

    # Initialisation store et résolveurs
    store = PartitionedCandidateStore(partitions_dir, max_cache_size=cache_size)
    config = RetrievalConfigV1(prefilter_k=prefilter_k, rescue_sidecar=True)
    tfidf_cache: OrderedDict = OrderedDict()
    persistent_cache = None  # In-memory RAM LRU only; avoids disk saturation on national scale
    memory_gov = MemoryGovernor(target_max_gb=max_ram_gb - 0.5, critical_max_gb=max_ram_gb, check_interval=10)
    spatial_resolver = SpatialResolver(ban_db_path) if ban_db_path.exists() else None

    power_mon = PowerMonitor(pause_on_lock=not no_pause_on_lock, idle_timeout_seconds=0.0)
    power_mon.start()

    # Initialisation CSV de détails
    fieldnames = [
        "crm_id", "crm_name", "gt_siret", "gt_etat",
        "gt_in_pool", "s1_hit1", "s1_hit50",
        "s2_hit1", "s2_hit5", "s2_top1_siret",
        "score_top1", "score_top2", "score_gap",
        "route_risk_score", "routing_decision", "routing_tier", "is_auto_correct",
        "name_jaro", "street_name_jaro", "postcode_match", "cand_is_active", "tie_breaker_applied",
    ]

    mode = "a" if is_resumed and details_path.exists() else "w"
    if mode == "a":
        try:
            with open(details_path, "r", encoding="utf-8", errors="replace") as f_chk:
                first_line = f_chk.readline()
                existing_cols = [c.strip() for c in first_line.split(";")]
                if "routing_tier" not in existing_cols:
                    fieldnames = [c for c in fieldnames if c != "routing_tier"]
        except Exception:
            pass

    details_file = open(details_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(details_file, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    # Métriques
    total_processed = len(processed_crm_ids)
    n_active = 0
    s1_act_hit1, s1_act_hit50 = 0, 0
    s2_act_hit1, s2_act_hit5 = 0, 0
    total_auto, correct_auto = 0, 0
    total_review = 0
    tier_a_count, tier_b_count = 0, 0

    if is_resumed and details_path.exists():
        try:
            with open(details_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    cid = safe_str(row.get("crm_id") or row.get("id"))
                    if cid not in processed_crm_ids:
                        continue
                    etat = safe_str(row.get("gt_etat") or "A").upper()
                    if etat == "A":
                        n_active += 1
                        s1_act_hit1 += int(row.get("s1_hit1") or 0)
                        s1_act_hit50 += int(row.get("s1_hit50") or 0)
                        s2_act_hit1 += int(row.get("s2_hit1") or 0)
                        s2_act_hit5 += int(row.get("s2_hit5") or 0)
                    r_dec = safe_str(row.get("routing_decision") or "REVIEW").upper()
                    if r_dec == "AUTO":
                        total_auto += 1
                        if int(row.get("is_auto_correct") or 0) == 1:
                            correct_auto += 1
                    else:
                        total_review += 1
            print(f"[REPRISE ATOMIQUE] Métriques historiques synchronisées : {n_active} actives, S1 Hit@1: {s1_act_hit1/max(n_active, 1)*100:.1f}%, S2 Hit@1: {s2_act_hit1/max(n_active, 1)*100:.1f}%, AUTO: {total_auto/max(total_processed, 1)*100:.1f}%")
        except Exception as e:
            logger.warning("Impossible de recharger les métriques historiques : %s", e)

    t0 = time.perf_counter()
    print("\n[3/4] Exécution du Benchmark End-to-End...")

    try:
        for idx, r in enumerate(eval_rows):
            crm_id = safe_str(r.get("crm_id") or r.get("id")) or str(idx)

            # Protocoles de sécurité Mobile Checkpointing & Closing Protection (Master Rule #9)
            if power_mon.should_exit():
                logger.warning("[SHUTDOWN] Signal d'extinction/fermeture OS détecté. Sauvegarde atomique et sortie propre...")
                details_file.flush()
                ckpt.step(total_processed, {})
                sys.exit(0)
            elif power_mon.should_pause():
                logger.info("[PAUSE] Mise en pause propre du batch (verrou/capot fermé/veille)...")
                details_file.flush()
                ckpt.step(total_processed, {})
                if hasattr(memory_gov, "release_memory"):
                    memory_gov.release_memory(store=store, tfidf_cache=tfidf_cache, reason="POWER_PAUSE")
                power_mon.wait_for_resume()
                if power_mon.should_exit():
                    logger.warning("[SHUTDOWN] Extinction système détectée pendant la pause. Sortie propre.")
                    sys.exit(0)
                logger.info("[REPRISE] Reprise transparente du batch après pause.")

            raw_gt = safe_str(r.get("gt_siret"))
            gt_siret = raw_gt.zfill(14) if raw_gt else ""
            gt_etat = safe_str(r.get("gt_etat") or r.get("sirene_etat") or "A").upper()
            is_active = (gt_etat == "A")

            crm_cp = safe_str(r.get("crm_cp") or r.get("postcode") or r.get("sirene_cp"))
            crm_insee = safe_str(r.get("crm_insee") or r.get("insee") or r.get("sirene_insee"))

            crm_row = {
                "crm_id": crm_id,
                "crm_name": safe_str(r.get("crm_name")),
                "crm_address": safe_str(r.get("crm_adresse") or r.get("crm_address")),
                "crm_city": safe_str(r.get("crm_commune") or r.get("crm_city")),
                "postcode": crm_cp.zfill(5) if crm_cp else "",
                "insee": crm_insee.zfill(5) if crm_insee else "",
            }
            crm_pre = preprocess_crm_row(crm_row)

            pool_res = build_candidate_pool(
                store=store,
                crm_row=crm_row,
                crm_pre=crm_pre,
                config=config,
                tfidf_cache=tfidf_cache,
                persistent_cache=persistent_cache,
                gt_siret=gt_siret,
                compute_idf_map=True,
            )

            candidates = pool_res.candidates or []
            cand_sirets = [str(c.get("siret") or "").strip().zfill(14) for c in candidates]
            gt_in_pool = (gt_siret in cand_sirets) if gt_siret else False

            s1_hit1, s1_hit50 = 0, 0
            s2_hit1, s2_hit5 = 0, 0
            s2_top1_siret = ""
            score_top1, score_top2, score_gap = 0.0, 0.0, 0.0
            route_risk_score = 0.0
            routing_decision = "REVIEW"
            routing_tier = ""
            is_auto_correct = 0
            top1_name_jaro = 0.0
            top1_street_jaro = 0.0
            top1_postcode_match = 0.0
            top1_cand_active = 0
            tie_breaker_applied = 0

            if candidates:
                # Stage 1 : Fast Ranker V9 (47 features)
                set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)
                base_matrix_fast, _ = make_features_batch(
                    crm_pre=crm_pre,
                    candidates=candidates,
                    skip_semantic=True,
                    spatial_resolver=spatial_resolver,
                )
                feat_fast = base_matrix_fast[:, fast_indices].astype(np.float32)
                dmat_fast = xgb.DMatrix(feat_fast, feature_names=FAST_RANKER_FEATURE_NAMES)
                s1_scores = ranker_booster.predict(dmat_fast)

                retained_indices, retained_sirets = apply_retention_topk(
                    candidates=candidates,
                    scores=s1_scores,
                    top_k=top_k,
                    feat_matrix=feat_fast,
                    feature_names=FAST_RANKER_FEATURE_NAMES,
                    crm_row=crm_row,
                    name_idf_map=pool_res.idf_map,
                )

                if gt_siret and gt_siret in retained_sirets:
                    s1_hit50 = 1
                    if retained_sirets[0] == gt_siret:
                        s1_hit1 = 1

                retained_cands = [candidates[i] for i in retained_indices]

                # Stage 2 : Unified Decider (67 features complètes avec NAF vectoriel)
                s2_matrix, _ = make_features_batch(
                    crm_pre=crm_pre,
                    candidates=retained_cands,
                    skip_semantic=False,
                    spatial_resolver=spatial_resolver,
                )
                dmat_s2 = xgb.DMatrix(s2_matrix, feature_names=FEATURE_NAMES)
                s2_raw_scores = decider_booster.predict(dmat_s2)

                if isotonic_calibrator is not None:
                    try:
                        s2_calibrated = isotonic_calibrator.predict_proba(s2_matrix)[:, 1]
                    except Exception:
                        s2_calibrated = s2_raw_scores
                else:
                    s2_calibrated = s2_raw_scores

                # Classement Stage 2 (Spécialiste 1 Ranker 1B si fourni, sinon Decider)
                if stage2_ranker_booster is not None:
                    s2_rank_scores = stage2_ranker_booster.predict(dmat_s2)
                    s2_order = list(np.argsort(-s2_rank_scores))
                else:
                    s2_order = list(np.argsort(-s2_calibrated))

                if use_tie_breaker:
                    s2_order, tie_meta = apply_stage2_tie_breakers(
                        retained_cands=retained_cands,
                        s2_scores=s2_calibrated,
                        s2_matrix=s2_matrix,
                        feature_names=FEATURE_NAMES,
                        max_k=5,
                        min_score_floor=0.25,
                        intra_siren_max_gap=0.20,
                        inter_siren_max_gap=0.06,
                    )
                    tie_breaker_applied = 1 if tie_meta.get("applied") else 0

                s2_top1_idx = s2_order[0]
                s2_top1_siret = str(retained_cands[s2_top1_idx].get("siret") or "").strip().zfill(14)
                score_top1 = float(s2_calibrated[s2_top1_idx])
                score_top2 = float(s2_calibrated[s2_order[1]]) if len(s2_order) > 1 else 0.0
                score_gap = score_top1 - score_top2

                if gt_siret:
                    ordered_sirets = [str(retained_cands[i].get("siret") or "").strip().zfill(14) for i in s2_order]
                    if gt_siret in ordered_sirets[:5]:
                        s2_hit5 = 1
                    if ordered_sirets[0] == gt_siret:
                        s2_hit1 = 1

                routing_tier = ""
                # Stage 3 : Routing
                if routing_mode == "deterministic":
                    det_res = route_from_feature_vector(
                        top1_features=s2_matrix[s2_top1_idx],
                        feature_names=FEATURE_NAMES,
                        score_top1=score_top1,
                        score_top2=score_top2,
                        candidate_dict=retained_cands[s2_top1_idx],
                        config=deterministic_config,
                    )
                    routing_decision = det_res.decision
                    routing_tier = det_res.tier or ""
                    route_risk_score = 1.0 if det_res.decision == "AUTO" else 0.0
                    top1_name_jaro = det_res.name_jaro
                    top1_street_jaro = det_res.street_name_jaro
                    top1_postcode_match = det_res.postcode_match
                    top1_cand_active = 1 if det_res.is_active else 0
                    if routing_decision == "AUTO":
                        if routing_tier == "TIER_A":
                            tier_a_count += 1
                        elif routing_tier == "TIER_B":
                            tier_b_count += 1
                elif risk_model is not None:
                    try:
                        # Construction rapide de la scène (top1, top2, deltas)
                        top1_feat = s2_matrix[s2_top1_idx]
                        top2_feat = s2_matrix[s2_order[1]] if len(s2_order) > 1 else np.zeros_like(top1_feat)
                        scene_dict = {
                            "score_top1": score_top1,
                            "score_top2": score_top2,
                            "score_gap": score_gap,
                            "score_ratio": score_top1 / max(score_top2, 0.001),
                        }
                        for fi, fn in enumerate(FEATURE_NAMES):
                            scene_dict[f"top1_{fn}"] = top1_feat[fi]
                            scene_dict[f"top2_{fn}"] = top2_feat[fi]
                            scene_dict[f"delta_{fn}"] = top1_feat[fi] - top2_feat[fi]

                        x_scene = np.array([[scene_dict.get(c, 0.0) for c in risk_feature_names]], dtype=np.float32)
                        if risk_calibrator is not None:
                            try:
                                risk_prob = float(risk_calibrator.predict_proba(x_scene)[:, 1][0])
                            except Exception:
                                risk_prob = float(risk_model.predict_proba(x_scene)[0, 1])
                        else:
                            risk_prob = float(risk_model.predict_proba(x_scene)[0, 1])
                        route_risk_score = risk_prob

                        if risk_prob >= risk_threshold:
                            routing_decision = "AUTO"
                        else:
                            routing_decision = "REVIEW"
                    except Exception:
                        routing_decision = "AUTO" if (score_top1 >= 0.85 and score_gap >= 0.20) else "REVIEW"
                else:
                    routing_decision = "AUTO" if (score_top1 >= 0.85 and score_gap >= 0.20) else "REVIEW"

                if routing_decision == "AUTO":
                    total_auto += 1
                    if s2_hit1 == 1:
                        correct_auto += 1
                        is_auto_correct = 1
                else:
                    total_review += 1

            total_processed += 1
            if is_active:
                n_active += 1
                s1_act_hit1 += s1_hit1
                s1_act_hit50 += s1_hit50
                s2_act_hit1 += s2_hit1
                s2_act_hit5 += s2_hit5

            # Écriture dans le CSV de détails
            row_dict = {
                "crm_id": crm_id,
                "crm_name": crm_row["crm_name"],
                "gt_siret": gt_siret,
                "gt_etat": gt_etat,
                "gt_in_pool": 1 if gt_in_pool else 0,
                "s1_hit1": s1_hit1,
                "s1_hit50": s1_hit50,
                "s2_hit1": s2_hit1,
                "s2_hit5": s2_hit5,
                "s2_top1_siret": s2_top1_siret,
                "score_top1": round(score_top1, 4),
                "score_top2": round(score_top2, 4),
                "score_gap": round(score_gap, 4),
                "route_risk_score": round(route_risk_score, 4),
                "routing_decision": routing_decision,
                "is_auto_correct": is_auto_correct,
                "name_jaro": round(top1_name_jaro, 4),
                "street_name_jaro": round(top1_street_jaro, 4),
                "postcode_match": round(top1_postcode_match, 1),
                "cand_is_active": top1_cand_active,
                "tie_breaker_applied": tie_breaker_applied,
            }
            if "routing_tier" in fieldnames:
                row_dict["routing_tier"] = routing_tier
            writer.writerow(row_dict)

            # Checkpoint & Memory release
            if len(tfidf_cache) > cache_size:
                tfidf_cache.popitem(last=False)
            memory_gov.check_and_release(store=store, tfidf_cache=tfidf_cache)

            if (idx + 1) % checkpoint_interval == 0 or (idx + 1) == len(eval_rows):
                details_file.flush()
                ckpt.step(total_processed, {})
                ckpt.save()
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / max(elapsed, 0.001)
                auto_prec = (correct_auto / total_auto * 100.0) if total_auto else 0.0
                auto_rate = (total_auto / total_processed * 100.0) if total_processed else 0.0
                print(
                    f"[{total_processed}/{len(all_rows)}] {rate:.1f} req/s | "
                    f"S1 Hit@1: {s1_act_hit1/max(n_active, 1)*100:.1f}% | "
                    f"S2 Hit@1: {s2_act_hit1/max(n_active, 1)*100:.1f}% | "
                    f"AUTO: {auto_rate:.1f}% @ {auto_prec:.2f}% Prec | "
                    f"RAM: {memory_gov.get_rss_gb():.2f} Go",
                    flush=True,
                )

        details_file.flush()
        details_file.close()

    finally:
        power_mon.stop()

    elapsed_total = time.perf_counter() - t0
    final_auto_prec = (correct_auto / total_auto * 100.0) if total_auto else 0.0
    final_auto_rate = (total_auto / total_processed * 100.0) if total_processed else 0.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "total_processed": total_processed,
        "active_queries": n_active,
        "elapsed_seconds": round(elapsed_total, 1),
        "stage1_active": {
            "hit1_rate": round(s1_act_hit1 / max(n_active, 1) * 100.0, 2),
            "hit50_rate": round(s1_act_hit50 / max(n_active, 1) * 100.0, 2),
        },
        "stage2_active": {
            "hit1_rate": round(s2_act_hit1 / max(n_active, 1) * 100.0, 2),
            "hit5_rate": round(s2_act_hit5 / max(n_active, 1) * 100.0, 2),
        },
        "stage3_routing": {
            "routing_mode": routing_mode,
            "auto_rate": round(final_auto_rate, 2),
            "auto_precision": round(final_auto_prec, 2),
            "total_auto": total_auto,
            "correct_auto": correct_auto,
            "total_review": total_review,
            "tier_a_count": tier_a_count,
            "tier_b_count": tier_b_count,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BILAN FINAL DU BENCHMARK NATIONAL 32K PIPE V7")
    print(f"Total Traité       : {total_processed} requêtes")
    print(f"Routing Mode S3    : {routing_mode.upper()}")
    print(f"Stage 1 Actif Hit@1: {summary['stage1_active']['hit1_rate']}% | Hit@50: {summary['stage1_active']['hit50_rate']}%")
    print(f"Stage 2 Actif Hit@1: {summary['stage2_active']['hit1_rate']}% | Hit@5: {summary['stage2_active']['hit5_rate']}%")
    print(f"Stage 3 AUTO Rate  : {final_auto_rate:.2f}% à {final_auto_prec:.2f}% de Précision Réelle")
    if routing_mode == "deterministic":
        print(f"  - Palier A (Haute certitude)      : {tier_a_count}")
        print(f"  - Palier B (Domination contextuelle): {tier_b_count}")
    print(f"Stage 3 REVIEW Rate: {100.0 - final_auto_rate:.2f}%")
    print(f"Rapports générés   : {summary_path} et {details_path}")
    print("=" * 80, flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate V7 Full Pipeline on 32k")
    parser.add_argument("--crm", type=Path, default=PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv")
    parser.add_argument("--partitions-dir", type=Path, default=PROJECT_ROOT / "data" / "candidates_v8_national")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "full_pipeline_32k")
    parser.add_argument("--tag", type=str, default="32k_v7_full_pipeline")
    parser.add_argument(
        "--routing-mode",
        type=str,
        choices=["deterministic", "risk"],
        default="deterministic",
        help="Stage 3 routing mode: 'deterministic' (3-Pillar Natural Selection) or 'risk' (legacy metamodel).",
    )
    parser.add_argument("--ranker-model", type=Path, default=PROJECT_ROOT / "models" / "xgbranker_fast_v9.json")
    parser.add_argument("--stage2-ranker-model", type=Path, default=None, help="Optional Stage 2 Ranker (1B ordering specialist)")
    parser.add_argument("--decider-model", type=Path, default=PROJECT_ROOT / "models" / "xgb_decider_unified_20261025_082618.json")
    parser.add_argument("--calibrator-path", type=Path, default=PROJECT_ROOT / "models" / "xgb_decider_unified_calibrator_isotonic_20261025_082618.pkl")
    parser.add_argument("--use-tie-breaker", action="store_true", default=False, help="Enable post-hoc Stage 2 Tie-Breaker Engine (default: OFF / Pure ML)")
    parser.add_argument("--risk-model", type=Path, default=PROJECT_ROOT / "models" / "routing_risk_model.pkl")
    parser.add_argument("--risk-calibrator", type=Path, default=PROJECT_ROOT / "models" / "routing_risk_calibrator.pkl")
    parser.add_argument("--risk-meta", type=Path, default=PROJECT_ROOT / "models" / "routing_risk_meta.json")
    parser.add_argument("--prefilter-k", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cache-size", type=int, default=40)
    parser.add_argument("--max-ram-gb", type=float, default=3.5)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--pause-on-lock", dest="no_pause_on_lock", action="store_false", help="Enable closing protection and auto-pause on lid close/lock.")
    parser.add_argument("--no-pause-on-lock", dest="no_pause_on_lock", action="store_true", default=False, help="Disable auto-pause on lid close/lock.")
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()

    run_pipeline_evaluation(
        crm_path=args.crm,
        partitions_dir=args.partitions_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        ranker_path=args.ranker_model,
        decider_path=args.decider_model,
        calibrator_path=args.calibrator_path,
        risk_model_path=args.risk_model,
        risk_calibrator_path=args.risk_calibrator,
        risk_meta_path=args.risk_meta,
        prefilter_k=args.prefilter_k,
        top_k=args.top_k,
        cache_size=args.cache_size,
        max_ram_gb=args.max_ram_gb,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        max_rows=args.max_rows,
        no_pause_on_lock=args.no_pause_on_lock,
        routing_mode=args.routing_mode,
        stage2_ranker_path=args.stage2_ranker_model,
        use_tie_breaker=args.use_tie_breaker,
    )


if __name__ == "__main__":
    main()
