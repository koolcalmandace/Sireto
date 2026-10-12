#!/usr/bin/env python3
"""
evaluate_all_lost_cases.py
==========================
Comprehensive evaluation of ALL 1,172 active cases previously lost (hit_50 == 0)
in the 32k benchmark, under:
  1) New Retrieval:
     - Partition fallback (load_by_insee_then_postcode when INSEE <= 2)
     - Address TF-IDF Top-50 strict inclusion union (up to 550 candidates)
     - Normalization & stopwords
  2) New Stage 1 Fast Ranker:
     - Sanitized retention rules (intrinsic signals: name_sim >= 0.70, jaro_enseigne >= 0.70)
"""

from __future__ import annotations
import csv, json, sys, time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.memory_governor import MemoryGovernor
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
from src.xgb_matcher.retention import apply_retention_topk


def cluster_rows_by_locality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _geo_key(r: Dict[str, Any]) -> tuple:
        insee = (r.get("crm_insee") or r.get("insee") or "").strip()
        cp = (r.get("crm_cp") or r.get("postcode") or "").strip()
        return (0 if insee else 1, insee or "99999", cp or "99999")

    return sorted(rows, key=_geo_key)


def main():
    print("=" * 80, flush=True)
    print("ÉVALUATION GLOBALE DES 1 172 CAS ACTIFS NON TROUVÉS (HIT@50 == 0)", flush=True)
    print("POST-NOUVEAU RETRIEVAL + POST-NOUVEAU STAGE 1 (FAST RANKER)", flush=True)
    print("=" * 80, flush=True)

    details_path = Path("reports/fast_ranker_32k/fast_ranker_32k_v8_unified_ban_details.csv")
    prev_info: Dict[str, Dict[str, Any]] = {}
    with open(details_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            if r.get("gt_etat") == "A" and int(r.get("hit_50", "0") or "0") == 0:
                cid = str(r.get("crm_id") or "").strip()
                in_pool = int(r.get("gt_in_pool", "0") or "0")
                reason = r.get("loss_reason", "")
                if in_pool == 1:
                    category = "685_ranker_pruned"
                elif reason == "NOT_IN_PARTITION":
                    category = "27_not_in_partition"
                elif reason == "PRUNED_BY_TFIDF":
                    category = "460_pruned_tfidf"
                else:
                    category = f"other_{reason}"
                prev_info[cid] = {
                    "crm_id": cid,
                    "category": category,
                    "prev_loss_reason": reason,
                    "prev_in_pool": in_pool,
                    "prev_raw_rank": int(r.get("raw_rank", "-1") or "-1"),
                }

    print(f"Total cas cibles identifiés : {len(prev_info)}", flush=True)
    cat_counts = {}
    for v in prev_info.values():
        c = v["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"Répartition par catégorie initiale : {cat_counts}", flush=True)

    # Recharger les données CRM complètes correspondantes
    crm_path = Path("data/crm_ok_gt_merged_v1.csv")
    matched_rows: List[Dict[str, str]] = []
    with open(crm_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            cid = (r.get("crm_id") or r.get("id") or "").strip()
            if cid in prev_info:
                matched_rows.append(r)

    print(f"Chargé {len(matched_rows)} lignes CRM correspondantes.", flush=True)

    # Clustérisation géographique pour optimiser la localité cache
    eval_rows = cluster_rows_by_locality(matched_rows)
    print(f"Clustérisation géographique terminée : {len(eval_rows)} requêtes ordonnées.", flush=True)

    # Initialisation store et modèle
    partitions_dir = Path("data/candidates_v8_national")
    store = PartitionedCandidateStore(partitions_dir, max_cache_size=40)
    governor = MemoryGovernor(target_max_gb=3.0, critical_max_gb=3.5, check_interval=10)

    model_path = Path("models/xgbranker_fast_v8.json")
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    config = RetrievalConfigV1(prefilter_k=500, rescue_sidecar=True)
    tfidf_cache: OrderedDict = OrderedDict()
    fast_indices = [FEATURE_NAMES.index(fn) for fn in FAST_RANKER_FEATURE_NAMES]

    t_start = time.perf_counter()
    n_total = len(eval_rows)

    stats_by_cat = {
        cat: {
            "total": 0,
            "in_pool": 0,
            "hit_1": 0,
            "hit_5": 0,
            "hit_10": 0,
            "hit_20": 0,
            "hit_50": 0,
        }
        for cat in set(cat_counts.keys())
    }

    results: List[Dict[str, Any]] = []

    for idx, r in enumerate(eval_rows):
        governor.check_and_release(store, tfidf_cache)

        crm_id = (r.get("crm_id") or r.get("id") or str(idx)).strip()
        crm_name = (r.get("crm_name") or r.get("Client final") or "").strip()
        crm_addr = (r.get("crm_adresse") or r.get("crm_address") or r.get("Adresse") or "").strip()
        crm_city = (r.get("crm_commune") or r.get("crm_city") or r.get("Commune") or "").strip()
        postcode = (r.get("crm_cp") or r.get("postcode") or r.get("Code Postal") or "").strip().zfill(5)
        insee = (r.get("crm_insee") or r.get("insee") or r.get("Code INSEE") or "").strip().zfill(5)
        gt_siret = (r.get("gt_siret") or r.get("ground_truth_siret") or "").strip().zfill(14)

        cat = prev_info[crm_id]["category"]
        stats_by_cat[cat]["total"] += 1

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

        if gt_in_pool:
            stats_by_cat[cat]["in_pool"] += 1
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

                stats_by_cat[cat]["hit_50"] += 1
                if retained_rank <= 20:
                    stats_by_cat[cat]["hit_20"] += 1
                if retained_rank <= 10:
                    stats_by_cat[cat]["hit_10"] += 1
                if retained_rank <= 5:
                    stats_by_cat[cat]["hit_5"] += 1
                if retained_rank == 1:
                    stats_by_cat[cat]["hit_1"] += 1

        results.append({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "gt_siret": gt_siret,
            "category": cat,
            "in_pool": gt_in_pool,
            "loss_reason": pool_res.loss_reason,
            "raw_rank": raw_rank,
            "retained_rank": retained_rank,
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "hit_20": hit_20,
            "hit_50": hit_50,
        })

        if len(tfidf_cache) > 40:
            tfidf_cache.popitem(last=False)

        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            elapsed = time.perf_counter() - t_start
            speed = (idx + 1) / max(elapsed, 0.001)
            tot_h50 = sum(s["hit_50"] for s in stats_by_cat.values())
            tot_pool = sum(s["in_pool"] for s in stats_by_cat.values())
            rss_gb = governor.get_rss_gb()
            print(
                f"[{idx+1}/{n_total}] {(idx+1)/n_total*100:.1f}% | {speed:.1f} req/s | "
                f"In-Pool: {tot_pool} | Hit@50: {tot_h50} ({tot_h50/(idx+1)*100:.1f}%) | RSS: {rss_gb:.2f} GB",
                flush=True,
            )

    elapsed_total = time.perf_counter() - t_start
    print("=" * 80, flush=True)
    print(f"ÉVALUATION TERMINÉE en {elapsed_total:.2f}s ({n_total/max(elapsed_total, 0.001):.1f} req/s)", flush=True)
    print("=" * 80, flush=True)

    print("\nRÉSULTATS DÉTAILLÉS PAR CATÉGORIE D'ORIGINE :", flush=True)
    for cat, s in stats_by_cat.items():
        print(f"\n--- Catégorie: {cat} (Total: {s['total']}) ---", flush=True)
        print(f"  In-Pool post-retrieval : {s['in_pool']} / {s['total']} ({s['in_pool']/max(s['total'],1)*100:.1f}%)", flush=True)
        print(f"  Hit@50 post-stage 1    : {s['hit_50']} / {s['total']} ({s['hit_50']/max(s['total'],1)*100:.1f}%)", flush=True)
        print(f"  Hit@20 post-stage 1    : {s['hit_20']} / {s['total']}", flush=True)
        print(f"  Hit@10 post-stage 1    : {s['hit_10']} / {s['total']}", flush=True)
        print(f"  Hit@5 post-stage 1     : {s['hit_5']} / {s['total']}", flush=True)
        print(f"  Hit@1 post-stage 1     : {s['hit_1']} / {s['total']}", flush=True)

    tot_all = n_total
    tot_pool_all = sum(s["in_pool"] for s in stats_by_cat.values())
    tot_h50_all = sum(s["hit_50"] for s in stats_by_cat.values())
    tot_h20_all = sum(s["hit_20"] for s in stats_by_cat.values())
    tot_h10_all = sum(s["hit_10"] for s in stats_by_cat.values())
    tot_h1_all = sum(s["hit_1"] for s in stats_by_cat.values())

    print("\n" + "=" * 80, flush=True)
    print("BILAN GLOBAL CONSOLIDÉ SUR LES 1 172 CAS ACTIFS PERDUS :", flush=True)
    print(f"  Total cas évalués        : {tot_all}", flush=True)
    print(f"  Réintégrés dans le Pool  : {tot_pool_all} / {tot_all} ({tot_pool_all/tot_all*100:.2f}%)", flush=True)
    print(f"  Nouveaux HIT@50 repêchés : {tot_h50_all} / {tot_all} ({tot_h50_all/tot_all*100:.2f}%)", flush=True)
    print(f"  Nouveaux HIT@20 repêchés : {tot_h20_all} / {tot_all} ({tot_h20_all/tot_all*100:.2f}%)", flush=True)
    print(f"  Nouveaux HIT@10 repêchés : {tot_h10_all} / {tot_all} ({tot_h10_all/tot_all*100:.2f}%)", flush=True)
    print(f"  Nouveaux HIT@1 repêchés  : {tot_h1_all} / {tot_all} ({tot_h1_all/tot_all*100:.2f}%)", flush=True)
    print("=" * 80, flush=True)

    # Enregistrer JSON et CSV
    out_json = Path("reports/fast_ranker_32k/fast_ranker_all_1172_lost_rescued.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_evaluated": tot_all,
            "elapsed_seconds": round(elapsed_total, 2),
            "speed_req_s": round(tot_all / max(elapsed_total, 0.001), 1),
            "overall": {
                "in_pool": tot_pool_all,
                "hit_50": tot_h50_all,
                "hit_20": tot_h20_all,
                "hit_10": tot_h10_all,
                "hit_1": tot_h1_all,
                "rescue_rate_pct": round(tot_h50_all / tot_all * 100, 2),
            },
            "by_category": stats_by_cat,
        }, f, indent=2)

    out_csv = Path("reports/fast_ranker_32k/fast_ranker_all_1172_lost_rescued_details.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crm_id", "crm_name", "gt_siret", "category", "in_pool",
            "loss_reason", "raw_rank", "retained_rank", "hit_1", "hit_5", "hit_10", "hit_20", "hit_50"
        ], delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    print(f"Rapports sauvegardés dans :\n  - {out_json}\n  - {out_csv}", flush=True)


if __name__ == "__main__":
    main()
