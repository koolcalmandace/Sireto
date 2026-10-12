#!/usr/bin/env python3
"""
evaluate_lost_685.py
====================
Targeted evaluation of the 685 active queries previously pruned at Stage 1
under the sanitized retention heuristics.
"""

from __future__ import annotations
import csv, json, sys, time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.features import (
    FAST_RANKER_FEATURE_NAMES,
    FEATURE_NAMES,
    make_features_batch,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.retention import apply_retention_topk, RetentionGuardConfig


def cluster_rows_by_locality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _geo_key(r: Dict[str, Any]) -> tuple:
        insee = (r.get("crm_insee") or r.get("insee") or "").strip()
        cp = (r.get("crm_cp") or r.get("postcode") or "").strip()
        return (0 if insee else 1, insee or "99999", cp or "99999")

    return sorted(rows, key=_geo_key)


def main():
    print("=" * 80, flush=True)
    print("ÉVALUATION CIBLÉE DES 685 CAS ACTIFS PRUNÉS PAR LE FAST RANKER", flush=True)
    print("=" * 80, flush=True)

    details_path = Path("reports/fast_ranker_32k/fast_ranker_32k_v8_unified_ban_details.csv")
    df_details = pd.read_csv(details_path, sep=";", low_memory=False)

    lost_df = df_details[(df_details["gt_etat"] == "A") & (df_details["gt_in_pool"] == 1) & (df_details["hit_50"] == 0)]
    lost_ids = set(lost_df["crm_id"].astype(str))
    print(f"Total cas cibles identifiés : {len(lost_ids)}", flush=True)

    # Recharger les données CRM complètes correspondantes
    crm_path = Path("data/crm_ok_gt_merged_v1.csv")
    matched_rows: List[Dict[str, str]] = []
    with open(crm_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            cid = (r.get("crm_id") or r.get("id") or "").strip()
            if cid in lost_ids:
                matched_rows.append(r)

    print(f"Chargé {len(matched_rows)} lignes CRM correspondantes.", flush=True)

    # Clustérisation géographique pour optimiser la localité cache
    eval_rows = cluster_rows_by_locality(matched_rows)
    print(f"Clustérisation géographique terminée : {len(eval_rows)} requêtes ordonnées.", flush=True)

    # Initialisation store et modèle
    partitions_dir = Path("data/candidates_v8_national")
    store = PartitionedCandidateStore(partitions_dir, max_cache_size=50)

    model_path = Path("models/xgbranker_fast_v8.json")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    config = RetrievalConfigV1(prefilter_k=500, rescue_sidecar=True)
    tfidf_cache: OrderedDict = OrderedDict()
    fast_indices = [FEATURE_NAMES.index(fn) for fn in FAST_RANKER_FEATURE_NAMES]

    t_start = time.perf_counter()
    n_total = len(eval_rows)
    recovered_hit50 = 0
    recovered_hit20 = 0
    recovered_hit10 = 0
    recovered_hit5 = 0
    recovered_hit1 = 0

    results: List[Dict[str, Any]] = []

    for idx, r in enumerate(eval_rows):
        crm_id = (r.get("crm_id") or r.get("id") or str(idx)).strip()
        crm_name = (r.get("crm_name") or r.get("Client final") or "").strip()
        crm_addr = (r.get("crm_adresse") or r.get("crm_address") or r.get("Adresse") or "").strip()
        crm_city = (r.get("crm_commune") or r.get("crm_city") or r.get("Commune") or "").strip()
        postcode = (r.get("crm_cp") or r.get("postcode") or r.get("Code Postal") or "").strip().zfill(5)
        insee = (r.get("crm_insee") or r.get("insee") or r.get("Code INSEE") or "").strip().zfill(5)
        gt_siret = (r.get("gt_siret") or r.get("ground_truth_siret") or "").strip().zfill(14)

        crm_row = {
            "crm_id": crm_id,
            "crm_name": crm_name,
            "crm_address": crm_addr,
            "crm_city": crm_city,
            "postcode": postcode,
            "insee": insee,
        }
        crm_pre = preprocess_crm_row(crm_row)

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
        cand_sirets = [str(c.get("siret") or "").strip().zfill(14) for c in candidates]
        gt_in_pool = gt_siret in cand_sirets

        raw_rank = -1
        retained_rank = -1
        hit_1 = 0
        hit_5 = 0
        hit_10 = 0
        hit_20 = 0
        hit_50 = 0

        if candidates and gt_in_pool:
            set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)
            base_matrix, _ = make_features_batch(
                crm_pre=crm_pre,
                candidates=candidates,
                skip_semantic=True,
            )
            feat_matrix = base_matrix[:, fast_indices].astype(np.float32)

            dmat = xgb.DMatrix(feat_matrix, feature_names=FAST_RANKER_FEATURE_NAMES)
            scores = booster.predict(dmat)
            pure_order = list(np.argsort(-scores))

            for r_pos, c_idx in enumerate(pure_order, start=1):
                if cand_sirets[c_idx] == gt_siret:
                    raw_rank = r_pos
                    break

            retained_indices, retained_sirets = apply_retention_topk(
                candidates=candidates,
                scores=scores,
                top_k=50,
                feat_matrix=feat_matrix,
                feature_names=FAST_RANKER_FEATURE_NAMES,
                crm_row=crm_row,
                name_idf_map=pool_res.idf_map,
            )

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

        recovered_hit1 += hit_1
        recovered_hit5 += hit_5
        recovered_hit10 += hit_10
        recovered_hit20 += hit_20
        recovered_hit50 += hit_50

        results.append({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "gt_siret": gt_siret,
            "raw_rank": raw_rank,
            "retained_rank": retained_rank,
            "hit_50": hit_50,
        })

        if len(tfidf_cache) > 50:
            tfidf_cache.popitem(last=False)

        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            elapsed = time.perf_counter() - t_start
            speed = (idx + 1) / max(elapsed, 0.001)
            print(
                f"[{idx+1}/{n_total}] {(idx+1)/n_total*100:.1f}% | Vitesse: {speed:.1f} req/s | "
                f"Repêchés Hit@50: {recovered_hit50}/{idx+1} ({recovered_hit50/(idx+1)*100:.1f}%)",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t_start
    summary = {
        "total_evaluated": n_total,
        "elapsed_seconds": round(elapsed_total, 2),
        "speed_req_s": round(n_total / max(elapsed_total, 0.001), 1),
        "hit_1": recovered_hit1,
        "hit_5": recovered_hit5,
        "hit_10": recovered_hit10,
        "hit_20": recovered_hit20,
        "hit_50": recovered_hit50,
        "recovery_rate_pct": round(recovered_hit50 / max(n_total, 1) * 100, 2),
    }

    out_path = Path("reports/fast_ranker_32k/fast_ranker_685_rescuetest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80, flush=True)
    print(f"TEST TERMINÉ en {elapsed_total:.2f}s ({n_total/max(elapsed_total, 0.001):.1f} req/s)", flush=True)
    print(f"Total cas évalués       : {n_total}", flush=True)
    print(f"Nouveaux Hit@50 repêchés : {recovered_hit50} / {n_total} ({summary['recovery_rate_pct']}%)", flush=True)
    print(f"Nouveaux Hit@10 repêchés : {recovered_hit10} / {n_total}", flush=True)
    print(f"Nouveaux Hit@1 repêchés  : {recovered_hit1} / {n_total}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
