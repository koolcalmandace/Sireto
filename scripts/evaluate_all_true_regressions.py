"""
evaluate_all_true_regressions.py — Standalone Certification Benchmark on the 93 Active Regressed Cases.

Evaluates the exact active cases where V8 had hit_50=1 and V9 had hit_50=0:
  - 19 false swap cases (raw_rank <= 50, brutally evicted by Guard 2C bug)
  - 49 near-miss cases (51 <= raw_rank <= 75)
  - 25 deep cases (raw_rank > 75)

Applies the Upper-Tier Sanctuary (ranks 1-35) and Rule F Near-Boundary Rescue in a single pass.
Generates an audit CSV and detailed metric breakdowns.
"""

from __future__ import annotations

import csv
import json
import time
import sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import xgboost as xgb

sys.path.insert(0, ".")
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import (
    FAST_RANKER_FEATURE_NAMES,
    FEATURE_NAMES,
    make_features_batch,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.retention import RetentionGuardConfig, apply_retention_topk

def main():
    print("=" * 80)
    print("BENCHMARK DE CERTIFICATION DES 93 CAS RÉGRESSÉS ACTIFS (POST-FIX)")
    print("=" * 80)

    # 1. Identify true active regressions (V8 hit_50=1, V9 hit_50=0, gt_etat='A')
    v8_hits = {}
    with open('reports/fast_ranker_32k/fast_ranker_32k_v8_retention_v2_details.csv', 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter=';'):
            v8_hits[r['crm_id']] = {
                'hit_50': (r.get('hit_50') == '1'),
                'retained_rank': int(r.get('retained_rank') or 999999),
                'raw_rank': int(r.get('raw_rank') or 999999),
            }

    crm_raw = {}
    with open('data/crm_ok_gt_merged_v1.csv', 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f, delimiter=';'):
            cid = (r.get('crm_id') or r.get('id') or '').strip()
            crm_raw[cid] = r

    target_cases = []
    with open('reports/fast_ranker_32k/fast_ranker_32k_v9_retention_v2_details.csv', 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter=';'):
            cid = r['crm_id']
            v8_info = v8_hits.get(cid, {'hit_50': False, 'retained_rank': 999999, 'raw_rank': 999999})
            v9_h50 = (r.get('hit_50') == '1')
            is_active = (r.get('gt_etat') == 'A')

            if is_active and v8_info['hit_50'] and not v9_h50:
                raw_crm = crm_raw.get(cid, {})
                target_cases.append({
                    'crm_id': cid,
                    'crm_name': (raw_crm.get('crm_name') or r.get('crm_name') or '').strip(),
                    'crm_adresse': (raw_crm.get('crm_adresse') or raw_crm.get('adresse') or '').strip(),
                    'crm_commune': (raw_crm.get('crm_commune') or raw_crm.get('city') or '').strip(),
                    'crm_cp': (raw_crm.get('crm_cp') or raw_crm.get('postcode') or r.get('crm_cp') or '').strip().zfill(5),
                    'crm_insee': (raw_crm.get('crm_insee') or raw_crm.get('insee') or r.get('crm_insee') or '').strip().zfill(5),
                    'gt_siret': (r.get('gt_siret') or '').strip().zfill(14),
                    'gt_etat': r.get('gt_etat', 'A'),
                    'v8_raw_rank': v8_info['raw_rank'],
                    'v8_retained_rank': v8_info['retained_rank'],
                    'v9_old_raw_rank': int(r.get('raw_rank') or 999999),
                    'v9_old_retained_rank': int(r.get('retained_rank') or 999999),
                    'v9_old_loss': r.get('loss_reason', ''),
                })

    print(f"Total cas actifs régressés identifiés : {len(target_cases)}")
    
    # Sort by locality for high-performance cached execution
    def _geo_key(x):
        return (x['crm_insee'], x['crm_cp'])
    target_cases.sort(key=_geo_key)

    # 2. Setup XGBoost V9 Model and Components
    bst = xgb.Booster()
    bst.load_model("models/xgbranker_fast_v9.json")
    fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

    store = PartitionedCandidateStore(Path("data/candidates_v8_national"), max_cache_size=100)
    config = RetrievalConfigV1(prefilter_k=500, rescue_sidecar=True)
    tfidf_cache = OrderedDict()

    retention_cfg = RetentionGuardConfig(
        top_k=50,
        max_swaps=8,
    )

    # 3. Execution
    results = []
    t0 = time.time()

    for idx, tc in enumerate(target_cases, 1):
        if idx % 10 == 0 or idx == len(target_cases):
            print(f"[{idx}/{len(target_cases)}] Traitement en cours... ({time.time()-t0:.1f}s)", flush=True)
        cid = tc['crm_id']
        gt_siret = tc['gt_siret']
        crm_row = {
            "crm_id": cid,
            "crm_name": tc['crm_name'],
            "crm_address": tc['crm_adresse'],
            "crm_city": tc['crm_commune'],
            "postcode": tc['crm_cp'],
            "insee": tc['crm_insee'],
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
        if not candidates:
            tc['v9_new_hit_50'] = 0
            tc['v9_new_raw_rank'] = -1
            tc['v9_new_retained_rank'] = 999999
            tc['v9_new_loss'] = "EMPTY_POOL"
            results.append(tc)
            continue

        cand_sirets = [str(c.get("siret") or "").strip().zfill(14) for c in candidates]
        gt_in_pool = (gt_siret in cand_sirets)

        set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)
        base_matrix, _ = make_features_batch(
            crm_pre=crm_pre,
            candidates=candidates,
            skip_semantic=True,
        )
        feat_matrix = base_matrix[:, fast_indices].astype(np.float32)
        dmat = xgb.DMatrix(feat_matrix, feature_names=FAST_RANKER_FEATURE_NAMES)
        scores = bst.predict(dmat)

        pure_order = list(np.argsort(-scores))
        new_raw_rank = 999999
        if gt_in_pool:
            for rank_pos, c_idx in enumerate(pure_order, start=1):
                if cand_sirets[c_idx] == gt_siret:
                    new_raw_rank = rank_pos
                    break

        retained_indices, retained_sirets = apply_retention_topk(
            candidates=candidates,
            scores=scores,
            top_k=50,
            config=retention_cfg,
            feat_matrix=feat_matrix,
            feature_names=FAST_RANKER_FEATURE_NAMES,
            crm_row=crm_row,
            name_idf_map=pool_res.idf_map,
        )

        new_retained_rank = 999999
        new_hit_50 = 0
        new_loss = ""

        if gt_in_pool:
            if gt_siret in retained_sirets:
                new_retained_rank = retained_sirets.index(gt_siret) + 1
                new_hit_50 = 1
            else:
                new_retained_rank = new_raw_rank
                new_loss = f"PRUNED_BY_FAST_RANKER_RANK_{new_raw_rank}"
        else:
            new_loss = pool_res.loss_reason or "NOT_IN_PREFILTER_POOL"

        tc['v9_new_hit_50'] = new_hit_50
        tc['v9_new_raw_rank'] = new_raw_rank
        tc['v9_new_retained_rank'] = new_retained_rank
        tc['v9_new_loss'] = new_loss
        results.append(tc)

    elapsed = time.time() - t0
    print(f"\nÉvaluation terminée en {elapsed:.1f} secondes ({len(results)} requêtes traitées).")

    # 4. Detailed Group Analysis
    # Group 1: False Swap Cases (v9_old_raw_rank <= 50)
    g1 = [r for r in results if 1 <= r['v9_old_raw_rank'] <= 50]
    g1_rescued = [r for r in g1 if r['v9_new_hit_50'] == 1]

    # Group 2: Near Misses (51 <= v9_old_raw_rank <= 75)
    g2 = [r for r in results if 51 <= r['v9_old_raw_rank'] <= 75]
    g2_rescued = [r for r in g2 if r['v9_new_hit_50'] == 1]

    # Group 3: Deep cases (v9_old_raw_rank > 75)
    g3 = [r for r in results if r['v9_old_raw_rank'] > 75 or r['v9_old_raw_rank'] == -1]
    g3_rescued = [r for r in g3 if r['v9_new_hit_50'] == 1]

    total_rescued = len(g1_rescued) + len(g2_rescued) + len(g3_rescued)

    print("\n" + "=" * 80)
    print("BILAN OFFICIEL DU BENCHMARK POST-FIX")
    print("=" * 80)
    print(f"Total cas régressés évalués : {len(results)}")
    print(f"Total cas sauvés et réintégrés dans le Top-50 : {total_rescued} / {len(results)} ({total_rescued/len(results)*100:.1f}%)")
    print(f"  - Groupe 1 (Sanctuaire Faux Swaps <= 50) : {len(g1_rescued)} / {len(g1)} récupérés ({len(g1_rescued)/len(g1)*100:.1f}%)")
    print(f"  - Groupe 2 (Near-Misses 51-75)          : {len(g2_rescued)} / {len(g2)} récupérés ({len(g2_rescued)/len(g2)*100:.1f}%)")
    print(f"  - Groupe 3 (Cas Profonds > 75 / Places) : {len(g3_rescued)} / {len(g3)} récupérés ({len(g3_rescued)/len(g3)*100:.1f}%)")

    print("\n--- ÉCHANTILLON GROUPE 1 RÉCUPÉRÉS PAR LE SANCTUAIRE (Rangs 1 à 35) ---")
    for r in g1_rescued[:10]:
        print(f"  [{r['crm_id']}] {r['crm_name'][:35] : <35} : ML_Raw={r['v9_new_raw_rank'] : <2} | V9_Old={r['v9_old_loss'] : <28} -> V9_New_Rank={r['v9_new_retained_rank']}")

    print("\n--- ÉCHANTILLON GROUPE 2 RÉCUPÉRÉS PAR LA RÈGLE F (Near-Boundary Rescue) ---")
    for r in g2_rescued[:10]:
        print(f"  [{r['crm_id']}] {r['crm_name'][:35] : <35} : ML_Raw={r['v9_new_raw_rank'] : <2} | V8_Rank={r['v8_retained_rank'] : <2} -> V9_New_Rank={r['v9_new_retained_rank']}")

    # Save details to CSV
    out_csv = Path("reports/fast_ranker_32k/fast_ranker_regressions_post_fix_details.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'crm_id', 'crm_name', 'crm_insee', 'crm_cp', 'gt_siret',
        'v8_retained_rank', 'v9_old_raw_rank', 'v9_old_loss',
        'v9_new_raw_rank', 'v9_new_retained_rank', 'v9_new_hit_50', 'v9_new_loss'
    ]
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFichier de détails exporté : {out_csv}")

if __name__ == "__main__":
    main()
