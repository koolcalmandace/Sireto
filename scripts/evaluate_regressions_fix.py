import csv, sys, time
from pathlib import Path
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

fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

# 1. Load V9 model
bst = xgb.Booster()
bst.load_model("models/xgbranker_fast_v9.json")

# 2. Load candidate store
store = PartitionedCandidateStore(Path("data/candidates_v8_national"))
config = RetrievalConfigV1()

# 3. Identify the 94 regressed cases
v8_ranks = {}
with open('reports/fast_ranker_32k/fast_ranker_32k_v8_retention_v2_details.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter=';'):
        cid = r['crm_id']
        r_str = r.get('retained_rank') or r.get('rank') or ''
        v8_ranks[cid] = int(r_str) if r_str.isdigit() else 999999

v9_regressions = []
with open('reports/fast_ranker_32k/fast_ranker_32k_v9_retention_v2_details.csv', 'r', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter=';'):
        cid = r['crm_id']
        r_str = r.get('retained_rank') or r.get('rank') or ''
        v9_r = int(r_str) if r_str.isdigit() else 999999
        v8_r = v8_ranks.get(cid, 999999)
        if v8_r <= 50 and v9_r > 50:
            v9_regressions.append((cid, v8_r, v9_r, r))

crm_raw = {}
with open('data/crm_ok_gt_merged_v1.csv', 'r', encoding='utf-8-sig') as f:
    for r in csv.DictReader(f, delimiter=';'):
        cid = (r.get('crm_id') or r.get('id') or '').strip()
        crm_raw[cid] = r

from collections import OrderedDict
tfidf_cache = OrderedDict()

print(f"Loaded {len(v9_regressions)} regressed cases to evaluate.", flush=True)

rescued_count = 0
results = []

t0 = time.time()
for idx, (cid, v8_r, v9_old_r, r) in enumerate(v9_regressions):
    if idx % 15 == 0:
        print(f"Processing query {idx+1}/{len(v9_regressions)}... (rescued so far: {rescued_count})", flush=True)
    raw = crm_raw.get(cid, {})
    gt_siret = (r.get('gt_siret') or '').strip().zfill(14)
    crm_name = (raw.get('crm_name') or r.get('crm_name') or '').strip()
    crm_addr = (raw.get('crm_adresse') or raw.get('adresse') or '').strip()
    postcode = (raw.get('crm_cp') or raw.get('postcode') or r.get('crm_cp') or '').strip().zfill(5)
    insee = (raw.get('crm_insee') or raw.get('insee') or r.get('crm_insee') or '').strip().zfill(5)
    
    crm_row = {
        "crm_id": cid,
        "crm_name": crm_name,
        "crm_address": crm_addr,
        "crm_city": raw.get('crm_commune') or '',
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
    if not candidates:
        continue
        
    set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)
    base_matrix, _ = make_features_batch(
        crm_pre=crm_pre,
        candidates=candidates,
        skip_semantic=True,
    )
    feat_matrix = base_matrix[:, fast_indices].astype(np.float32)
    dmat = xgb.DMatrix(feat_matrix, feature_names=FAST_RANKER_FEATURE_NAMES)
    scores = bst.predict(dmat)
    
    retained_indices, retained_sirets = apply_retention_topk(
        candidates=candidates,
        scores=scores,
        top_k=50,
        feat_matrix=feat_matrix,
        feature_names=FAST_RANKER_FEATURE_NAMES,
        crm_row=crm_row,
        name_idf_map=pool_res.idf_map,
    )
    
    new_rank = 999999
    if gt_siret in retained_sirets:
        new_rank = retained_sirets.index(gt_siret) + 1
        
    is_rescued = (new_rank <= 50)
    if is_rescued:
        rescued_count += 1
        
    results.append({
        'cid': cid,
        'name': crm_name,
        'v8_rank': v8_r,
        'v9_old_rank': v9_old_r,
        'v9_new_rank': new_rank,
        'is_rescued': is_rescued,
    })

elapsed = time.time() - t0
print(f"\nCompleted evaluation in {elapsed:.1f}s.")
print(f"Rescued into Top-50: {rescued_count} / {len(v9_regressions)} ({rescued_count/len(v9_regressions)*100:.1f}%)")

# Group 1 Check
print(f"\n--- Group 1 False Swap Fixes Check (Upper-Tier Sanctuary) ---")
g1_cids = ['FR016286', 'FR019149', 'AC019614', 'FR026366', 'FR026365', '118318']
for res in results:
    if res['cid'] in g1_cids:
        print(f"  [{res['cid']}] {res['name']}: v8={res['v8_rank']}, v9_old={res['v9_old_rank']} -> v9_new={res['v9_new_rank']} (Rescued: {res['is_rescued']})")

# Group 2 Check
g2 = [res for res in results if 51 <= res['v9_old_rank'] <= 75]
g2_rescued = [res for res in g2 if res['is_rescued']]
print(f"\n--- Group 2 Near Misses Check (51-75): {len(g2_rescued)} / {len(g2)} rescued ({len(g2_rescued)/len(g2)*100:.1f}%) ---")
for res in g2_rescued[:10]:
    print(f"  [{res['cid']}] {res['name']}: v8={res['v8_rank']}, v9_old={res['v9_old_rank']} -> v9_new={res['v9_new_rank']}")
