"""
train_fast_ranker_model.py
==========================
Train the Stage 1 Fast Ranker model (XGBoost) with updated 38 fast features:
- Includes is_siege (headquarters signal)
- Includes addr_unsupported_by_name (address co-tenant penalty)
- Decontaminated _is_sidecar_match (genuine sidecar anchors only)
- Safe batch standard with MemoryGovernor, LRU bounded cache
- Outputs models/xgbranker_fast_v8.json
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.checkpoint import TrainingFeatureCheckpointManager
from src.xgb_matcher.features import (
    FAST_RANKER_FEATURE_NAMES,
    FEATURE_NAMES,
    make_features_batch,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.memory_governor import MemoryGovernor
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgbranker_fast_v8.json"
META_PATH = MODELS_DIR / "xgbranker_fast_v8_meta.json"


def train_fast_ranker(
    crm_path: Path = PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv",
    partitions_dir: Path = PROJECT_ROOT / "data" / "candidates_v8_national",
    sample_size: int = 3000,
    random_seed: int = 42,
    num_boost_round: int = 80,
    resume: bool = True,
    chunk_size: int = 500,
    checkpoint_dir: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
) -> xgb.Booster:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("ENTRAÎNEMENT DU FAST RANKER V8 (STAGE 1)")
    print(f"Features ({len(FAST_RANKER_FEATURE_NAMES)}) : {FAST_RANKER_FEATURE_NAMES}")
    print(f"Échantillon d'entraînement : {sample_size} requêtes (seed={random_seed})")
    print(f"Atomic Checkpointing : {'Actif (chunk=' + str(chunk_size) + ')' if resume else 'Désactivé'}")
    print("=" * 80)

    # 1. Chargement & découpage GroupKFold par SIREN
    df_crm = pd.read_csv(crm_path, sep=";", dtype=str)
    df_valid = df_crm[df_crm["gt_siret"].notna() & (df_crm["gt_siret"].str.strip().str.len() == 14)].copy()
    df_valid["siren"] = df_valid["gt_siret"].str.strip().str[:9]

    # Échantillonnage clusterisé par commune pour maximiser le hit-rate Parquet et TF-IDF
    communes = [c for c in df_valid["crm_insee"].dropna().unique() if c.strip()]
    rng = np.random.RandomState(random_seed)
    shuffled_communes = rng.permutation(communes)

    selected_indices = []
    for comm in shuffled_communes:
        c_idxs = df_valid.index[df_valid["crm_insee"] == comm].tolist()
        selected_indices.extend(c_idxs)
        if len(selected_indices) >= sample_size:
            break

    sample_df = df_valid.loc[selected_indices[:sample_size]].copy().reset_index(drop=True)

    gkf = GroupKFold(n_splits=3)
    train_idx, dev_idx = next(gkf.split(sample_df, groups=sample_df["siren"]))
    sample_df["split"] = "train"
    sample_df.iloc[dev_idx, sample_df.columns.get_loc("split")] = "dev"

    # Tri géographique pour cache Parquet optimal
    sample_df = sample_df.sort_values(["crm_insee", "crm_cp"]).reset_index(drop=True)

    # 2. Extraction des pools et features avec Checkpointing Atomique
    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "data" / "train_checkpoints" / f"fast_ranker_{sample_size}_seed{random_seed}"

    ckpt_mgr = TrainingFeatureCheckpointManager(
        tag=f"fast_ranker_{sample_size}_seed{random_seed}",
        checkpoint_dir=checkpoint_dir,
        sample_size=sample_size,
        random_seed=random_seed,
        feature_names=FAST_RANKER_FEATURE_NAMES,
        chunk_size=chunk_size,
    )

    total = len(sample_df)
    start_idx = 0
    if resume and ckpt_mgr.can_resume():
        (
            last_idx,
            train_X_list,
            train_y_list,
            train_groups,
            dev_X_list,
            dev_y_list,
            dev_groups,
        ) = ckpt_mgr.load_checkpoint()
        start_idx = last_idx + 1
        print(f"\n[REPRISE ATOMIQUE] Checkpoint détecté dans {checkpoint_dir}")
        print(f"  Reprise immédiate à l'index {start_idx}/{total} (Train={len(train_groups)}, Dev={len(dev_groups)})\n")
    else:
        if not resume:
            ckpt_mgr.clear()
        train_X_list, train_y_list, train_groups = [], [], []
        dev_X_list, dev_y_list, dev_groups = [], [], []

    store = PartitionedCandidateStore(partitions_dir, max_cache_size=40)
    config = RetrievalConfigV1(prefilter_k=500, rescue_sidecar=True)
    tfidf_cache: OrderedDict = OrderedDict()
    persistent_cache = TfidfPersistentCache(PROJECT_ROOT / "data" / "tfidf_cache")
    memory_gov = MemoryGovernor(target_max_gb=3.0, critical_max_gb=3.8, check_interval=10)

    fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

    t0 = time.perf_counter()
    print("\nExtraction des features pour l'apprentissage...")

    for idx in range(start_idx, total):
        row = sample_df.iloc[idx]
        gt_siret = str(row["gt_siret"]).strip().zfill(14)
        cp = str(row.get("crm_cp") or "").strip().zfill(5)
        insee = str(row.get("crm_insee") or "").strip().zfill(5)

        crm_row = {
            "crm_name": str(row.get("crm_name") or "").strip(),
            "crm_address": str(row.get("crm_adresse") or row.get("crm_address") or "").strip(),
            "crm_city": str(row.get("crm_commune") or row.get("crm_city") or "").strip(),
            "postcode": cp,
            "insee": insee,
            "crm_id": str(row.get("crm_id") or str(idx)),
        }
        crm_pre = preprocess_crm_row(crm_row)

        try:
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
        except Exception:
            continue

        candidates = pool_res.candidates
        if not candidates:
            continue

        cands_dict = {str(c.get("siret") or "").strip().zfill(14): c for c in candidates if "siret" in c}
        gt_in_pool = gt_siret in cands_dict
        if not gt_in_pool:
            continue

        cand_list = list(cands_dict.values())
        set_global_name_idf_map(pool_res.idf_map, pool_res.default_idf)

        base_matrix, _ = make_features_batch(
            crm_pre=crm_pre,
            candidates=cand_list,
            skip_semantic=True,
        )

        feat_matrix = base_matrix[:, fast_indices].astype(np.float32)
        y = np.array([1.0 if str(c.get("siret") or "").strip().zfill(14) == gt_siret else 0.0 for c in cand_list], dtype=np.float32)
        split = row["split"]

        if split == "train":
            train_X_list.append(feat_matrix)
            train_y_list.append(y)
            train_groups.append(len(cand_list))
        else:
            dev_X_list.append(feat_matrix)
            dev_y_list.append(y)
            dev_groups.append(len(cand_list))

        ckpt_mgr.add_sample(
            idx=idx,
            split=split,
            feat_matrix=feat_matrix,
            y=y,
            group_size=len(cand_list),
        )

        if len(tfidf_cache) > 40:
            tfidf_cache.popitem(last=False)
        memory_gov.check_and_release(store=store, tfidf_cache=tfidf_cache)

        if (idx + 1) % 250 == 0 or (idx + 1) == total:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1 - start_idx) / max(elapsed, 0.001)
            print(f"  Extraction : {idx + 1}/{total} ({rate:.1f} req/s) [RAM: {memory_gov.get_rss_gb():.2f} Go]", flush=True)

    ckpt_mgr.finalize(last_idx=total - 1)
    print(f"Extraction terminée en {time.perf_counter() - t0:.1f}s.", flush=True)
    print(f"  Train: {len(train_groups)} requêtes, Dev: {len(dev_groups)} requêtes.", flush=True)

    # 3. Entraînement XGBoost Pairwise
    print("\nEntraînement XGBoost Ranker...")
    X_train = np.vstack(train_X_list)
    y_train = np.concatenate(train_y_list)
    X_dev = np.vstack(dev_X_list)
    y_dev = np.concatenate(dev_y_list)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FAST_RANKER_FEATURE_NAMES)
    dtrain.set_group(train_groups)
    ddev = xgb.DMatrix(X_dev, label=y_dev, feature_names=FAST_RANKER_FEATURE_NAMES)
    ddev.set_group(dev_groups)

    params = {
        "objective": "rank:pairwise",
        "tree_method": "hist",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": ["ndcg@5", "ndcg@50"],
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (ddev, "dev")],
        verbose_eval=20,
    )

    # 4. Sauvegarde
    target_model_path = output_model_path or MODEL_PATH
    target_meta_path = target_model_path.with_name(target_model_path.stem + "_meta.json")

    # Backup existing model if present
    if target_model_path.exists():
        bak_path = target_model_path.with_name(target_model_path.stem + ".bak.json")
        try:
            import shutil
            shutil.copy2(target_model_path, bak_path)
        except Exception:
            pass

    booster.save_model(str(target_model_path))
    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": target_model_path.name,
                "feature_names": FAST_RANKER_FEATURE_NAMES,
                "num_features": len(FAST_RANKER_FEATURE_NAMES),
                "params": params,
                "train_queries": len(train_groups),
                "dev_queries": len(dev_groups),
                "num_boost_round": num_boost_round,
            },
            f,
            indent=2,
        )

    print(f"\n[SUCCÈS] Modèle Fast Ranker V8 sauvegardé dans : {target_model_path}")
    return booster


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fast Ranker V8 Model")
    parser.add_argument("--sample-size", type=int, default=3000, help="Number of queries for training/dev")
    parser.add_argument("--rounds", type=int, default=80, help="Number of boosting rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", default=False, help="Disable auto-resume and start fresh")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of queries per atomic chunk")
    parser.add_argument("--output-model", type=str, default=None, help="Custom output model path")
    args = parser.parse_args()

    train_fast_ranker(
        sample_size=args.sample_size,
        num_boost_round=args.rounds,
        random_seed=args.seed,
        resume=not args.no_resume,
        chunk_size=args.chunk_size,
        output_model_path=Path(args.output_model) if args.output_model else None,
    )
