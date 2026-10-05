#!/usr/bin/env python3
"""
evaluate_fast_ranker_32k.py
===========================
Comprehensive Stage 1 Fast Ranker Evaluator on the National 32K Dataset.
Evaluates candidate pool retention from K=500 (Retrieval) to K=50 (Fast Ranker).

Features evaluated:
  - 38 Fast Features (including is_siege and addr_unsupported_by_name)
  - Decontaminated Sidecar match flag
  - Top-50 selection with Siège Social & Sidecar Anchor retention safeguards

Key Safety Features:
  - Atomic Checkpointing & Auto-Resume (--resume)
  - MemoryGovernor (RAM < 3.5 GB)
  - PowerMonitor (Windows sleep / lid-close / lock event detection)
  - In-Python geographic clustering for optimal Parquet cache locality (>99% hit rate)

Outputs:
  - reports/fast_ranker_32k/fast_ranker_32k_summary.json
  - reports/fast_ranker_32k/fast_ranker_32k_details.csv
  - Atomic checkpoints in reports/fast_ranker_32k/.checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.checkpoint import CheckpointManager
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
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_fast_ranker_32k")


def load_crm_gt(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
    return rows


def cluster_rows_by_locality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _geo_key(r: Dict[str, Any]) -> tuple:
        insee = (r.get("crm_insee") or r.get("insee") or "").strip()
        cp = (r.get("crm_cp") or r.get("postcode") or "").strip()
        return (0 if insee else 1, insee or "99999", cp or "99999")

    return sorted(rows, key=_geo_key)


def apply_retention_topk(
    candidates: List[Dict[str, Any]],
    scores: np.ndarray,
    top_k: int = 50,
) -> Tuple[List[int], List[str]]:
    """
    Ranks candidates by XGBoost score and applies retention safeguards:
    1. Siège Social preservation: if a SIREN is already selected in top_k, ensure its
       head office establishment is included in the top_k budget.
    2. Sidecar Anchor preservation: if a candidate was rescued via genuine Sidecar match,
       ensure it is retained in the top_k budget.
    Returns:
        retained_indices: list of candidate indices in final rank order (length <= top_k)
        retained_sirets: list of 14-digit SIRETs
    """
    n_cands = len(candidates)
    if n_cands <= top_k:
        order = list(np.argsort(-scores))
        return order, [str(candidates[i].get("siret") or "").strip().zfill(14) for i in order]

    order = list(np.argsort(-scores))
    selected_indices = list(order[:top_k])
    selected_sirets = [str(candidates[i].get("siret") or "").strip().zfill(14) for i in selected_indices]
    selected_sirens = {s[:9] for s in selected_sirets if len(s) >= 9}

    # Inspect candidates ranked between top_k and min(n_cands, 200) for retention candidates
    candidates_to_protect = []
    for rank_idx in range(top_k, min(n_cands, 200)):
        cand_idx = order[rank_idx]
        c = candidates[cand_idx]
        s = str(c.get("siret") or "").strip().zfill(14)
        siren = s[:9]

        is_siege = bool(c.get("is_siege", False))
        is_sidecar = bool(c.get("_is_sidecar_match") or c.get("is_sidecar_match"))

        # Rule 1: Siège Social of an already active SIREN
        if is_siege and siren in selected_sirens:
            candidates_to_protect.append((cand_idx, "SIEGE_RETENTION"))
            continue

        # Rule 2: Validated Sidecar match
        if is_sidecar:
            candidates_to_protect.append((cand_idx, "SIDECAR_RETENTION"))
            continue

    if not candidates_to_protect:
        return selected_indices, selected_sirets

    # Safely swap protected candidates into the bottom of the budget,
    # replacing lower-ranked secondary annexes of multi-site SIRENs
    final_indices = list(selected_indices)
    protected_added = 0

    for cand_idx, reason in candidates_to_protect:
        if protected_added >= 5:  # Cap retention replacements to max 5 slots
            break
        # Find candidate from the bottom of the top_k that is not a siege and not a sidecar
        replace_slot = None
        for slot in range(top_k - 1, 0, -1):
            cur_idx = final_indices[slot]
            cur_c = candidates[cur_idx]
            cur_siege = bool(cur_c.get("is_siege", False))
            cur_sidecar = bool(cur_c.get("_is_sidecar_match") or cur_c.get("is_sidecar_match"))
            if not cur_siege and not cur_sidecar:
                replace_slot = slot
                break

        if replace_slot is not None:
            final_indices[replace_slot] = cand_idx
            protected_added += 1

    final_sirets = [str(candidates[i].get("siret") or "").strip().zfill(14) for i in final_indices]
    return final_indices, final_sirets


def evaluate(
    rows: List[Dict[str, str]],
    partitions_dir: Path,
    output_dir: Path,
    tag: str,
    model_path: Path,
    prefilter_k: int = 500,
    top_k: int = 50,
    cache_size: int = 40,
    max_ram_gb: float = 3.5,
    checkpoint_interval: int = 250,
    resume: bool = True,
    fresh: bool = False,
    log_file: Optional[Path] = None,
    enable_power_monitor: bool = True,
    pause_on_lock: bool = True,
    idle_timeout_seconds: float = 300.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / ".checkpoints"

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info("LANCEMENT DU BENCHMARK FAST RANKER 32K (STAGE 1)")
    logger.info("Modèle : %s", model_path)
    logger.info("Features (%d) : %s", len(FAST_RANKER_FEATURE_NAMES), FAST_RANKER_FEATURE_NAMES)
    logger.info("Dataset : %d requêtes", len(rows))
    logger.info("Paramètres : prefilter_k=%d, top_k=%d, max_ram_gb=%.1f", prefilter_k, top_k, max_ram_gb)
    logger.info("=" * 80)

    # 1. Chargement modèle
    if not model_path.exists():
        logger.info("Modèle %s non trouvé. Entraînement automatique préalable...", model_path)
        from scripts.train_fast_ranker_model import train_fast_ranker
        booster = train_fast_ranker(partitions_dir=partitions_dir)
    else:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        logger.info("Modèle Fast Ranker chargé avec succès.")

    # 2. Initialisation des composants
    memory_gov = MemoryGovernor(
        target_max_gb=max_ram_gb,
        critical_max_gb=max_ram_gb + 0.5,
        check_interval=10,
    )
    store = PartitionedCandidateStore(partitions_dir, max_cache_size=cache_size)
    config = RetrievalConfigV1(
        prefilter_k=prefilter_k,
        rescue_sidecar=True,
    )
    tfidf_cache: OrderedDict = OrderedDict()
    fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

    # Checkpoint management
    ckpt = CheckpointManager(
        tag=tag,
        checkpoint_dir=ckpt_dir,
        total_rows=len(rows),
        crm_path=Path("data/crm_ok_gt_merged_v1.csv"),
        config_signature=config.signature().hash,
        save_interval=checkpoint_interval,
    )

    details_path = output_dir / f"fast_ranker_{tag}_details.csv"
    detail_fieldnames = [
        "crm_id",
        "crm_name",
        "crm_insee",
        "crm_cp",
        "gt_siret",
        "gt_etat",
        "pool_size",
        "gt_in_pool",
        "raw_rank",
        "retained_rank",
        "hit_1",
        "hit_5",
        "hit_10",
        "hit_20",
        "hit_50",
        "loss_reason",
    ]

    processed_crm_ids: Set[str] = set()
    is_resumed = False
    if resume and not fresh:
        processed_crm_ids = ckpt.sync_from_details_csv(details_path, detail_fieldnames)
        is_resumed = len(processed_crm_ids) > 0

    ckpt.init_details_writer(details_path, detail_fieldnames, resume=is_resumed)

    if processed_crm_ids:
        unprocessed_rows = [
            r for r in rows
            if (r.get("crm_id") or r.get("id") or "").strip() not in processed_crm_ids
        ]
        logger.info(
            "Reprise : %d / %d requêtes déjà traitées (%.1f%%). Reste : %d requêtes.",
            len(processed_crm_ids),
            len(rows),
            (len(processed_crm_ids) / max(len(rows), 1)) * 100,
            len(unprocessed_rows),
        )
    else:
        unprocessed_rows = list(rows)

    eval_rows = cluster_rows_by_locality(unprocessed_rows)
    logger.info("Clustering géographique terminé : %d requêtes ordonnées.", len(eval_rows))

    power_mon = PowerMonitor(
        pause_on_lock=pause_on_lock,
        idle_timeout_seconds=idle_timeout_seconds,
    )

    # Accumulateurs de métriques
    total_processed = len(processed_crm_ids)
    n_active = 0
    n_closed = 0

    # Rétention Active
    act_gt_in_pool = 0
    act_hit1 = 0
    act_hit5 = 0
    act_hit10 = 0
    act_hit20 = 0
    act_hit50 = 0

    # Global
    all_gt_in_pool = 0
    all_hit1 = 0
    all_hit5 = 0
    all_hit10 = 0
    all_hit20 = 0
    all_hit50 = 0

    if is_resumed and details_path.exists():
        try:
            with open(details_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    cid = (row.get("crm_id") or "").strip()
                    if cid not in processed_crm_ids:
                        continue
                    etat = (row.get("gt_etat") or "A").strip().upper()
                    gt_in_p = int(row.get("gt_in_pool") or 0)
                    h1 = int(row.get("hit_1") or 0)
                    h5 = int(row.get("hit_5") or 0)
                    h10 = int(row.get("hit_10") or 0)
                    h20 = int(row.get("hit_20") or 0)
                    h50 = int(row.get("hit_50") or 0)

                    if etat == "A":
                        n_active += 1
                        if gt_in_p:
                            act_gt_in_pool += 1
                        act_hit1 += h1
                        act_hit5 += h5
                        act_hit10 += h10
                        act_hit20 += h20
                        act_hit50 += h50
                    else:
                        n_closed += 1

                    all_gt_in_pool += gt_in_p
                    all_hit1 += h1
                    all_hit5 += h5
                    all_hit10 += h10
                    all_hit20 += h20
                    all_hit50 += h50
            logger.info("Métriques synchronisées depuis l'historique : %d requêtes rechargées (%d actives).", len(processed_crm_ids), n_active)
        except Exception as e:
            logger.warning("Impossible de relire les métriques historiques : %s", e)

    t_start = time.perf_counter()
    logger.info("Démarrage de l'évaluation...")

    for i, r in enumerate(eval_rows):
        if enable_power_monitor:
            power_mon.check_power_events(
                on_pause=lambda: [ckpt.flush_csv(), gc.collect()],
                on_resume=lambda: logger.info("Reprise après événement PowerGuard."),
            )

        crm_id = (r.get("crm_id") or r.get("id") or str(i)).strip()
        crm_name = (r.get("crm_name") or r.get("Client final") or "").strip()
        crm_addr = (r.get("crm_adresse") or r.get("crm_address") or r.get("Adresse") or "").strip()
        crm_city = (r.get("crm_commune") or r.get("crm_city") or r.get("Commune") or "").strip()
        postcode = (r.get("crm_cp") or r.get("postcode") or r.get("Code Postal") or "").strip().zfill(5)
        insee = (r.get("crm_insee") or r.get("insee") or r.get("Code INSEE") or "").strip().zfill(5)
        gt_siret = (r.get("gt_siret") or r.get("ground_truth_siret") or "").strip().zfill(14)
        gt_etat = (r.get("sirene_etat") or r.get("etat_admin") or "A").strip().upper()

        crm_row = {
            "crm_id": crm_id,
            "crm_name": crm_name,
            "crm_address": crm_addr,
            "crm_city": crm_city,
            "postcode": postcode,
            "insee": insee,
        }
        crm_pre = preprocess_crm_row(crm_row)

        # 1. Retrieval K=500
        pool_res = build_candidate_pool(
            store=store,
            crm_row=crm_row,
            crm_pre=crm_pre,
            config=config,
            tfidf_cache=tfidf_cache,
            gt_siret=gt_siret,
            compute_idf_map=True,
        )

        candidates = pool_res.candidates or []
        pool_size = len(candidates)
        cand_sirets = [str(c.get("siret") or "").strip().zfill(14) for c in candidates]
        gt_in_pool = gt_siret in cand_sirets if gt_siret else False

        raw_rank = -1
        retained_rank = -1
        hit_1 = 0
        hit_5 = 0
        hit_10 = 0
        hit_20 = 0
        hit_50 = 0
        loss_reason = ""

        if candidates:
            # 2. Fast Ranker Feature Computation (38 features)
            set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)
            base_matrix, _ = make_features_batch(
                crm_pre=crm_pre,
                candidates=candidates,
                skip_semantic=True,
            )
            feat_matrix = base_matrix[:, fast_indices].astype(np.float32)

            # 3. XGBoost Scoring
            dmat = xgb.DMatrix(feat_matrix, feature_names=FAST_RANKER_FEATURE_NAMES)
            scores = booster.predict(dmat)

            # 4. Pure XGBoost Ranking
            pure_order = list(np.argsort(-scores))
            if gt_in_pool:
                for rank_pos, c_idx in enumerate(pure_order, start=1):
                    if cand_sirets[c_idx] == gt_siret:
                        raw_rank = rank_pos
                        break

            # 5. Top-K Retention (Headquarters + Sidecar Protection)
            retained_indices, retained_sirets = apply_retention_topk(
                candidates=candidates,
                scores=scores,
                top_k=top_k,
            )

            if gt_in_pool:
                if gt_siret in retained_sirets:
                    retained_rank = retained_sirets.index(gt_siret) + 1
                    hit_50 = 1
                    if retained_rank <= 20:
                        hit_20 = 1
                    if retained_rank <= 10:
                        hit_10 = 1
                    if retained_rank <= 5:
                        hit_5 = 1
                    if retained_rank == 1:
                        hit_1 = 1
                else:
                    retained_rank = raw_rank
                    loss_reason = f"PRUNED_BY_FAST_RANKER_RANK_{raw_rank}"
        else:
            loss_reason = pool_res.loss_reason or "EMPTY_POOL"

        if not gt_in_pool and not loss_reason:
            loss_reason = pool_res.loss_reason or "NOT_IN_PREFILTER_POOL"

        # Mise à jour compteurs
        total_processed += 1
        is_active = (gt_etat == "A")
        if is_active:
            n_active += 1
            if gt_in_pool:
                act_gt_in_pool += 1
            act_hit1 += hit_1
            act_hit5 += hit_5
            act_hit10 += hit_10
            act_hit20 += hit_20
            act_hit50 += hit_50
        else:
            n_closed += 1

        all_gt_in_pool += int(gt_in_pool)
        all_hit1 += hit_1
        all_hit5 += hit_5
        all_hit10 += hit_10
        all_hit20 += hit_20
        all_hit50 += hit_50

        # Enregistrement CSV détail
        ckpt.record_detail({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "crm_insee": insee,
            "crm_cp": postcode,
            "gt_siret": gt_siret,
            "gt_etat": gt_etat,
            "pool_size": pool_size,
            "gt_in_pool": int(gt_in_pool),
            "raw_rank": raw_rank,
            "retained_rank": retained_rank,
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "hit_20": hit_20,
            "hit_50": hit_50,
            "loss_reason": loss_reason if hit_50 == 0 else "",
        })

        if len(tfidf_cache) > 40:
            tfidf_cache.popitem(last=False)
        memory_gov.check_and_release(store=store, tfidf_cache=tfidf_cache)

        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(eval_rows):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / max(elapsed, 0.001)
            rss_gb = memory_gov.get_rss_gb()
            ckpt.step(crm_id, {})
            ckpt.flush_csv()

            act_ret = (act_hit50 / max(act_gt_in_pool, 1)) * 100
            act_rec = (act_hit50 / max(n_active, 1)) * 100

            logger.info(
                "[%d/%d] %.1f%% | Vitesse: %.1f req/s | RAM: %.2f Go | Actifs Hit@50: %.2f%% (Rétention: %.2f%%)",
                total_processed,
                len(rows),
                (total_processed / max(len(rows), 1)) * 100,
                rate,
                rss_gb,
                act_rec,
                act_ret,
            )

    # Synthèse Finale
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "total_queries": total_processed,
        "active_queries": n_active,
        "closed_queries": n_closed,
        "metrics_active": {
            "retrieval_pool_recall": (act_gt_in_pool / max(n_active, 1)) * 100,
            "fast_ranker_hit1": (act_hit1 / max(n_active, 1)) * 100,
            "fast_ranker_hit5": (act_hit5 / max(n_active, 1)) * 100,
            "fast_ranker_hit10": (act_hit10 / max(n_active, 1)) * 100,
            "fast_ranker_hit20": (act_hit20 / max(n_active, 1)) * 100,
            "fast_ranker_hit50": (act_hit50 / max(n_active, 1)) * 100,
            "retention_rate_from_pool": (act_hit50 / max(act_gt_in_pool, 1)) * 100,
        },
        "metrics_all": {
            "retrieval_pool_recall": (all_gt_in_pool / max(total_processed, 1)) * 100,
            "fast_ranker_hit1": (all_hit1 / max(total_processed, 1)) * 100,
            "fast_ranker_hit5": (all_hit5 / max(total_processed, 1)) * 100,
            "fast_ranker_hit10": (all_hit10 / max(total_processed, 1)) * 100,
            "fast_ranker_hit20": (all_hit20 / max(total_processed, 1)) * 100,
            "fast_ranker_hit50": (all_hit50 / max(total_processed, 1)) * 100,
            "retention_rate_from_pool": (all_hit50 / max(all_gt_in_pool, 1)) * 100,
        },
    }

    summary_path = output_dir / f"fast_ranker_{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info("BENCHMARK TERMINÉ AVEC SUCCÈS")
    logger.info("Rapport résumé : %s", summary_path)
    logger.info("Rapport détail : %s", details_path)
    logger.info("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Évaluation Fast Ranker 32K (Stage 1)")
    parser.add_argument("--crm", type=Path, default=Path("data/crm_ok_gt_merged_v1.csv"))
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v8_national"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fast_ranker_32k"))
    parser.add_argument("--tag", type=str, default="32k_v8_retention")
    parser.add_argument("--model-path", type=Path, default=Path("models/xgbranker_fast_v8.json"))
    parser.add_argument("--prefilter-k", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cache-size", type=int, default=40)
    parser.add_argument("--max-ram-gb", type=float, default=3.5)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", action="store_true", default=False)
    parser.add_argument("--log-file", type=Path, default=Path("logs/fast_ranker_32k.log"))
    args = parser.parse_args()

    rows = load_crm_gt(args.crm)
    evaluate(
        rows=rows,
        partitions_dir=args.partitions_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        model_path=args.model_path,
        prefilter_k=args.prefilter_k,
        top_k=args.top_k,
        cache_size=args.cache_size,
        max_ram_gb=args.max_ram_gb,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume and not args.fresh,
        fresh=args.fresh,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
