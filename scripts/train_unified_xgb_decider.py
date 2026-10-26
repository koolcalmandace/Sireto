#!/usr/bin/env python3
"""
scripts/train_unified_xgb_decider.py
======================================
Unified Stage 2 XGBoost Decider Training Pipeline.

Key Architectural Guarantees:
  1. Feature Invariance (Stage Inversion Solved):
     F_Stage1 (47 features) ⊂ F_Stage2 (68 features). All Stage 1 signals
     (continuous BAN GPS, exact address anchors, local authority flags, acronyms)
     are strictly preserved and visible to Stage 2.
  2. Semantic Atomic Coder:
     Activates AtomicSemanticCoder (384-dimensional continuous character-wb n-gram
     vector space) and continuous NAF projection without heavy PyTorch dependencies.
  3. Clean Stop & Resilience:
     Monitors Windows power events (PowerMonitor) and traps Ctrl+C (SIGINT).
     Flushes pending queries atomically into compressed .npz chunks + manifest.json.
     Supports instant resume (--resume) with 0 duplicated compute.
  4. GroupKFold on SIREN (Levier 1):
     Strict 70% Train / 15% Dev / 15% Test partitioning by SIREN. Zero data leakage.
  5. Realistic Inference Distribution:
     Stage 1 Fast Ranker V9 selects Top-50 candidates per query (K=50).
  6. Organic Margin Separation & Calibration (Levier 5):
     Trained with binary:logistic and calibrated via Isotonic Regression on Dev split.
"""

from __future__ import annotations

import os
import sys

# Critical: activate semantic mode before importing feature modules
os.environ["XGB_SEMANTIC_ENABLED"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import gc
import json
import logging
import pickle
import signal
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
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
from src.xgb_matcher.power_guard import PowerMonitor
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.semantic import AtomicSemanticCoder, is_semantic_available
from src.xgb_matcher.spatial import SpatialResolver
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache

logger = logging.getLogger("train_unified_xgb_decider")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Calibrator wrapper classes (pickle-compatible with inference pipeline)
class IsotonicCalibrator:
    """Wrapper for isotonic calibration of a base XGBoost classifier."""

    def __init__(self, base_estimator: Any, iso_reg: Any) -> None:
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_estimator, "predict_proba"):
            proba = self.base_estimator.predict_proba(X)[:, 1]
        else:
            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            proba = self.base_estimator.predict(dmat)
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SigmoidCalibrator:
    """Wrapper for sigmoid (Platt) calibration of a base XGBoost classifier."""

    def __init__(self, base_estimator: Any, lr: Any) -> None:
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.base_estimator, "predict_proba"):
            proba = self.base_estimator.predict_proba(X)[:, 1]
        else:
            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            proba = self.base_estimator.predict(dmat)
        return self.lr.predict_proba(proba.reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 50,
    "eval_metric": ["auc", "logloss"],
    "seed": 42,
    "n_estimators": 400,
}

EARLY_STOPPING_ROUNDS = 30


def compute_hit_at_k(y_true: np.ndarray, y_pred: np.ndarray, groups: List[int], k: int = 1) -> float:
    """Compute Hit@k across grouped query blocks."""
    if not groups:
        return 0.0
    hits = 0
    start = 0
    for group_size in groups:
        end = start + group_size
        g_labels = y_true[start:end]
        g_scores = y_pred[start:end]
        top_indices = np.argsort(-g_scores)[:k]
        if any(g_labels[i] == 1.0 for i in top_indices):
            hits += 1
        start = end
    return hits / len(groups)


def train_unified_decider(
    crm_path: Path = PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv",
    partitions_dir: Path = PROJECT_ROOT / "data" / "candidates_v8_national",
    ranker_model_path: Path = PROJECT_ROOT / "models" / "xgbranker_fast_v9.json",
    ban_db_path: Path = PROJECT_ROOT / "data" / "ban.duckdb",
    sample_size: int = 10000,
    random_seed: int = 42,
    chunk_size: int = 250,
    top_k: int = 50,
    resume: bool = True,
    output_dir: Path = PROJECT_ROOT / "models",
    checkpoint_dir: Optional[Path] = None,
    calibration_method: str = "isotonic",
    unattended: bool = False,
) -> Dict[str, Any]:
    """Execute unified Stage 2 XGBoost decider feature extraction and training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("SIRETO V7 — ENTRAÎNEMENT DU XGB DECIDER UNIFIÉ (STAGE 2)")
    print(f"Features unifiées ({len(FEATURE_NAMES)}) : {FEATURE_NAMES}")
    print(f"Sous-ensemble Stage 1 vérifié : {set(FAST_RANKER_FEATURE_NAMES).issubset(set(FEATURE_NAMES))}")
    print(f"Échantillon cible : {sample_size} requêtes (seed={random_seed})")
    print(f"Moteur sémantique atomique : {AtomicSemanticCoder.__name__} (384-dim continuous)")
    print(f"Clean Stop & Power Guard : Actifs (Verrouillage / Inactivité / SIGINT Ctrl+C)")
    print(f"Top-K filtrage Stage 1 : {top_k} candidats par requête")
    print("=" * 80)

    # 1. Chargement & découpage GroupKFold(siren) 70 / 15 / 15
    print("\n[1/5] Chargement du CRM et découpage GroupKFold par SIREN (70/15/15)...")
    df_crm = pd.read_csv(crm_path, sep=";", dtype=str)
    df_valid = df_crm[df_crm["gt_siret"].notna() & (df_crm["gt_siret"].str.strip().str.len() == 14)].copy()
    df_valid["siren"] = df_valid["gt_siret"].str.strip().str[:9]

    # Échantillonnage clusterisé par commune pour optimiser les accès disque
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

    # Découpage strict en 3 partitions par SIREN: Train (70%), Dev (15%), Test (15%)
    unique_sirens = sample_df["siren"].unique()
    rng_siren = np.random.RandomState(random_seed)
    shuffled_sirens = rng_siren.permutation(unique_sirens)
    n_sirens = len(shuffled_sirens)

    n_train = int(n_sirens * 0.70)
    n_dev = int(n_sirens * 0.15)

    train_sirens = set(shuffled_sirens[:n_train])
    dev_sirens = set(shuffled_sirens[n_train : n_train + n_dev])
    test_sirens = set(shuffled_sirens[n_train + n_dev :])

    sample_df["split"] = "train"
    sample_df.loc[sample_df["siren"].isin(dev_sirens), "split"] = "dev"
    sample_df.loc[sample_df["siren"].isin(test_sirens), "split"] = "test"

    # Vérification stricte d'étanchéité (Zéro Data Leakage)
    sirens_train = set(sample_df[sample_df["split"] == "train"]["siren"])
    sirens_dev = set(sample_df[sample_df["split"] == "dev"]["siren"])
    sirens_test = set(sample_df[sample_df["split"] == "test"]["siren"])
    assert len(sirens_train.intersection(sirens_dev)) == 0, "Data leakage détecté entre Train et Dev!"
    assert len(sirens_train.intersection(sirens_test)) == 0, "Data leakage détecté entre Train et Test!"
    assert len(sirens_dev.intersection(sirens_test)) == 0, "Data leakage détecté entre Dev et Test!"

    counts = sample_df["split"].value_counts().to_dict()
    print(f"  Répartition des requêtes : Train={counts.get('train', 0)}, Dev={counts.get('dev', 0)}, Test={counts.get('test', 0)}")

    # Tri géographique pour cache Parquet optimal
    sample_df = sample_df.sort_values(["crm_insee", "crm_cp"]).reset_index(drop=True)

    # 2. Initialisation du Checkpointing Atomique et du Clean Stop
    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "data" / "train_checkpoints" / f"unified_decider_h1max_{sample_size}_seed{random_seed}"

    ckpt_mgr = TrainingFeatureCheckpointManager(
        tag=f"unified_decider_h1max_{sample_size}_seed{random_seed}",
        checkpoint_dir=checkpoint_dir,
        sample_size=sample_size,
        random_seed=random_seed,
        feature_names=FEATURE_NAMES,
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
            test_X_list,
            test_y_list,
            test_groups,
        ) = ckpt_mgr.load_checkpoint(include_test=True)
        start_idx = last_idx + 1
        print(f"\n[REPRISE ATOMIQUE] Checkpoint détecté dans {checkpoint_dir}")
        print(f"  Reprise immédiate à l'index {start_idx}/{total}")
        print(f"  Progression restaurée : Train={len(train_groups)}, Dev={len(dev_groups)}, Test={len(test_groups)} requêtes\n")
    else:
        if not resume:
            ckpt_mgr.clear()
        train_X_list, train_y_list, train_groups = [], [], []
        dev_X_list, dev_y_list, dev_groups = [], [], []
        test_X_list, test_y_list, test_groups = [], [], []

    # Initialisation des composants d'inférence et de ressources
    print("\n[2/5] Initialisation du Fast Ranker V9, store de candidats et résolveur BAN...")
    ranker_booster = xgb.Booster()
    ranker_booster.load_model(str(ranker_model_path))

    store = PartitionedCandidateStore(partitions_dir, max_cache_size=40)
    config = RetrievalConfigV1(prefilter_k=500, rescue_sidecar=True)
    tfidf_cache: OrderedDict = OrderedDict()
    persistent_cache = TfidfPersistentCache(PROJECT_ROOT / "data" / "tfidf_cache")
    memory_gov = MemoryGovernor(target_max_gb=3.0, critical_max_gb=3.8, check_interval=10)

    spatial_resolver = None
    if ban_db_path.exists():
        spatial_resolver = SpatialResolver(ban_db_path)

    fast_indices = [FEATURE_NAMES.index(name) for name in FAST_RANKER_FEATURE_NAMES]

    # Enregistrement du gestionnaire de Clean Stop (SIGINT Ctrl+C / PowerMonitor)
    stop_requested = False

    def clean_stop_handler(sig: int, frame: Any) -> None:
        nonlocal stop_requested
        print("\n" + "=" * 80)
        print("[CLEAN STOP] Signal d'interruption intercepté. Finalisation de la requête active...")
        print("=" * 80, flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, clean_stop_handler)
    try:
        signal.signal(signal.SIGTERM, clean_stop_handler)
    except Exception:
        pass

    if unattended:
        print("\n[POWER GUARD] Mode UNATTENDED actif: calcul continu sans pause sur inactivité/verrouillage.")
        power_mon = PowerMonitor(pause_on_lock=False, idle_timeout_seconds=0.0)
    else:
        power_mon = PowerMonitor(pause_on_lock=True, idle_timeout_seconds=300.0)
    power_mon.start()

    t0 = time.perf_counter()
    print("\n[3/5] Extraction des pools Top-50 et features unifiées 68D avec Atomic Checkpoint...")

    try:
        for idx in range(start_idx, total):
            # Check Clean Stop avant de commencer la requête
            if stop_requested or power_mon.should_exit():
                print(f"\n[CLEAN STOP] Sauvegarde atomique du checkpoint à l'index {idx - 1}...")
                ckpt_mgr.flush_chunk(last_idx=max(0, idx - 1))
                memory_gov.release_memory(store=store, tfidf_cache=tfidf_cache, reason="CLEAN_STOP")
                print("=" * 80)
                print(f"ÉTAT SAUVEGARDÉ SANS PERTE DANS LE CHECKPOINT ATOMIQUE (index {idx - 1}/{total})")
                print(f"Pour reprendre l'entraînement :")
                print(f"  python scripts/train_unified_xgb_decider.py --sample-size {sample_size} --resume")
                print("=" * 80, flush=True)
                sys.exit(0)
            elif power_mon.should_pause():
                print(f"\n[PAUSE SYSTÈME] Verrouillage / inactivité détecté. Sauvegarde du chunk à l'index {idx - 1}...")
                ckpt_mgr.flush_chunk(last_idx=max(0, idx - 1))
                memory_gov.release_memory(store=store, tfidf_cache=tfidf_cache, reason="POWER_PAUSE")
                power_mon.wait_for_resume()
                if power_mon.should_exit() or stop_requested:
                    sys.exit(0)

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

            # Étape 1 : Fast Ranker V9 scoring (47 features)
            base_matrix_fast, _ = make_features_batch(
                crm_pre=crm_pre,
                candidates=cand_list,
                skip_semantic=True,
                spatial_resolver=spatial_resolver,
            )
            feat_fast = base_matrix_fast[:, fast_indices].astype(np.float32)
            dmat_fast = xgb.DMatrix(feat_fast, feature_names=FAST_RANKER_FEATURE_NAMES)
            fast_scores = ranker_booster.predict(dmat_fast)

            # Filtrage Top-K (K=50) + préservation GT
            order = np.argsort(-fast_scores)
            selected_indices = list(order[:top_k])

            gt_cand_idx = next((i for i, c in enumerate(cand_list) if str(c.get("siret") or "").strip().zfill(14) == gt_siret), None)
            if gt_cand_idx is not None and gt_cand_idx not in selected_indices:
                selected_indices.append(gt_cand_idx)

            retained_cands = [cand_list[i] for i in selected_indices]

            # Étape 2 : Extraction complète des 68 features Stage 2 avec AtomicSemanticCoder
            stage2_matrix, _ = make_features_batch(
                crm_pre=crm_pre,
                candidates=retained_cands,
                skip_semantic=False,
                spatial_resolver=spatial_resolver,
            )

            y = np.array(
                [1.0 if str(c.get("siret") or "").strip().zfill(14) == gt_siret else 0.0 for c in retained_cands],
                dtype=np.float32,
            )
            split = row["split"]

            if split == "train":
                train_X_list.append(stage2_matrix)
                train_y_list.append(y)
                train_groups.append(len(retained_cands))
            elif split == "test":
                test_X_list.append(stage2_matrix)
                test_y_list.append(y)
                test_groups.append(len(retained_cands))
            else:
                dev_X_list.append(stage2_matrix)
                dev_y_list.append(y)
                dev_groups.append(len(retained_cands))

            ckpt_mgr.add_sample(
                idx=idx,
                split=split,
                feat_matrix=stage2_matrix,
                y=y,
                group_size=len(retained_cands),
            )

            if len(tfidf_cache) > 40:
                tfidf_cache.popitem(last=False)
            memory_gov.check_and_release(store=store, tfidf_cache=tfidf_cache)

            if (idx + 1) % 250 == 0 or (idx + 1) == total:
                elapsed = time.perf_counter() - t0
                rate = (idx + 1 - start_idx) / max(elapsed, 0.001)
                print(
                    f"  Extraction : {idx + 1}/{total} ({rate:.1f} req/s) "
                    f"[Train={len(train_groups)}, Dev={len(dev_groups)}, Test={len(test_groups)}] "
                    f"[RAM: {memory_gov.get_rss_gb():.2f} Go]",
                    flush=True,
                )

        ckpt_mgr.finalize(last_idx=total - 1)
        print(f"\nExtraction terminée en {time.perf_counter() - t0:.1f}s.", flush=True)
    finally:
        power_mon.stop()

    # 4. Entraînement du XGBoost Decider binaire
    print("\n[4/5] Entraînement du XGBoost Decider binaire avec pondération colocataires...")
    X_train = np.vstack(train_X_list)
    y_train = np.concatenate(train_y_list)

    X_dev = np.vstack(dev_X_list)
    y_dev = np.concatenate(dev_y_list)

    X_test = np.vstack(test_X_list) if test_X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y_test = np.concatenate(test_y_list) if test_y_list else np.empty((0,), dtype=np.float32)

    print(f"  Dimensions matrices : Train={X_train.shape}, Dev={X_dev.shape}, Test={X_test.shape}")
    print(f"  Positifs : Train={int(y_train.sum())}, Dev={int(y_dev.sum())}, Test={int(y_test.sum())}")

    # -------------------------------------------------------------------------
    # Hard Negative Mining : surpondération ciblée des imposteurs du Fast Ranker
    # -------------------------------------------------------------------------
    sample_weights = np.ones(len(X_train), dtype=np.float32)
    addr_jaro_idx = FEATURE_NAMES.index("addr_jaro")
    name_jaro_idx = FEATURE_NAMES.index("name_jaro_max")
    is_siege_idx = FEATURE_NAMES.index("is_siege")
    delta_siege_idx = FEATURE_NAMES.index("delta_addr_vs_siege")
    geo_exact_idx = FEATURE_NAMES.index("geo_exact_match")
    exact_anchor_idx = FEATURE_NAMES.index("exact_address_anchor")

    # 1. Imposteurs Top-3 du Fast Ranker (candidats classés rang 1, 2 ou 3 mais y == 0.0)
    top3_neg_mask = np.zeros(len(X_train), dtype=bool)
    curr = 0
    for g in train_groups:
        for offset in range(min(3, g)):
            if y_train[curr + offset] == 0.0:
                top3_neg_mask[curr + offset] = True
        curr += g
    sample_weights[top3_neg_mask] = np.maximum(sample_weights[top3_neg_mask], 3.0)

    # 2. Sièges distants déceptifs (siège officiel tentant de voler le match au local)
    remote_siege_mask = (X_train[:, is_siege_idx] >= 0.5) & (X_train[:, delta_siege_idx] < -0.2) & (y_train == 0.0)
    sample_weights[remote_siege_mask] = np.maximum(sample_weights[remote_siege_mask], 3.5)

    # 3. Colocataires négatifs (adresse identique mais nom distant)
    coloc_mask = (X_train[:, addr_jaro_idx] > 0.80) & (X_train[:, name_jaro_idx] < 0.65) & (y_train == 0.0)
    sample_weights[coloc_mask] = np.maximum(sample_weights[coloc_mask], 3.0)

    # 4. Homonymes hors commune (nom élevé mais mauvaise localisation)
    homonym_far_mask = (X_train[:, name_jaro_idx] > 0.85) & (X_train[:, geo_exact_idx] == 0.0) & (y_train == 0.0)
    sample_weights[homonym_far_mask] = np.maximum(sample_weights[homonym_far_mask], 3.0)

    # 5. Imposteurs d'ancre d'adresse (adresse exacte mais entreprise différente)
    anchor_impostor_mask = (X_train[:, exact_anchor_idx] == 1.0) & (X_train[:, name_jaro_idx] < 0.60) & (y_train == 0.0)
    sample_weights[anchor_impostor_mask] = np.maximum(sample_weights[anchor_impostor_mask], 3.5)

    total_hard_negatives = int(np.sum(sample_weights > 1.0))
    print(f"  Hard Negative Mining appliqué dans le Train :")
    print(f"    - Imposteurs Top-3 Fast Ranker (3.0x) : {int(top3_neg_mask.sum())}")
    print(f"    - Sièges distants déceptifs (3.5x)    : {int(remote_siege_mask.sum())}")
    print(f"    - Colocataires même adresse (3.0x)    : {int(coloc_mask.sum())}")
    print(f"    - Homonymes hors commune (3.0x)       : {int(homonym_far_mask.sum())}")
    print(f"    - Imposteurs d'ancre d'adresse (3.5x) : {int(anchor_impostor_mask.sum())}")
    print(f"    - Total lignes négatives surpondérées : {total_hard_negatives} ({total_hard_negatives / len(X_train) * 100:.1f}%)")

    classifier = xgb.XGBClassifier(
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **CLASSIFIER_PARAMS,
    )
    classifier.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_dev, y_dev)],
        verbose=25,
    )

    # 5. Calibration Isotonique sur le Dev set (Levier 5)
    print("\n[5/5] Calibration Isotonique des probabilités et Évaluation des Métriques...")
    calibrator = None
    if calibration_method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        y_dev_proba = classifier.predict_proba(X_dev)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_dev_proba, y_dev)
        calibrator = IsotonicCalibrator(classifier, iso)
    elif calibration_method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        y_dev_proba = classifier.predict_proba(X_dev)[:, 1]
        lr = LogisticRegression()
        lr.fit(y_dev_proba.reshape(-1, 1), y_dev)
        calibrator = SigmoidCalibrator(classifier, lr)

    def evaluate_split(split_name: str, X: np.ndarray, y: np.ndarray, groups: List[int], model: Any) -> Dict[str, Any]:
        if len(y) == 0:
            return {}
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(xgb.DMatrix(X))
        auc = float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
        ap = float(average_precision_score(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
        brier = float(brier_score_loss(y, y_prob)) if len(np.unique(y)) > 1 else 0.0
        hit1 = compute_hit_at_k(y, y_prob, groups, k=1) * 100.0
        hit3 = compute_hit_at_k(y, y_prob, groups, k=3) * 100.0
        hit5 = compute_hit_at_k(y, y_prob, groups, k=5) * 100.0

        # Calcul de la séparation organique de marge (score_gap top1 - top2)
        score_gaps = []
        st = 0
        for g_sz in groups:
            en = st + g_sz
            g_scores = y_prob[st:en]
            sorted_s = np.sort(g_scores)[::-1]
            gap = float(sorted_s[0] - sorted_s[1]) if len(sorted_s) > 1 else float(sorted_s[0])
            score_gaps.append(gap)
            st = en

        mean_gap = float(np.mean(score_gaps)) if score_gaps else 0.0
        median_gap = float(np.median(score_gaps)) if score_gaps else 0.0

        return {
            "split": split_name,
            "auc": round(auc, 5),
            "average_precision": round(ap, 5),
            "brier_score_loss": round(brier, 5),
            "hit_at_1": round(hit1, 2),
            "hit_at_3": round(hit3, 2),
            "hit_at_5": round(hit5, 2),
            "mean_score_gap": round(mean_gap, 4),
            "median_score_gap": round(median_gap, 4),
        }

    dev_metrics_raw = evaluate_split("dev_raw", X_dev, y_dev, dev_groups, classifier)
    test_metrics_raw = evaluate_split("test_raw", X_test, y_test, test_groups, classifier)

    dev_metrics_calib = evaluate_split("dev_calibrated", X_dev, y_dev, dev_groups, calibrator) if calibrator else {}
    test_metrics_calib = evaluate_split("test_calibrated", X_test, y_test, test_groups, calibrator) if calibrator else {}

    print("\n" + "=" * 80)
    print("BILAN DES PERFORMANCES DU STAGE 2 DECIDER UNIFIÉ")
    print("=" * 80)
    print(f"Dev  (Brut)       : Hit@1={dev_metrics_raw.get('hit_at_1')}% | AUC={dev_metrics_raw.get('auc')} | Brier={dev_metrics_raw.get('brier_score_loss')} | Marge Médiane={dev_metrics_raw.get('median_score_gap')}")
    if dev_metrics_calib:
        print(f"Dev  (Calibré)    : Hit@1={dev_metrics_calib.get('hit_at_1')}% | AUC={dev_metrics_calib.get('auc')} | Brier={dev_metrics_calib.get('brier_score_loss')} | Marge Médiane={dev_metrics_calib.get('median_score_gap')}")
    print(f"Test (Brut)       : Hit@1={test_metrics_raw.get('hit_at_1')}% | AUC={test_metrics_raw.get('auc')} | Brier={test_metrics_raw.get('brier_score_loss')} | Marge Médiane={test_metrics_raw.get('median_score_gap')}")
    if test_metrics_calib:
        print(f"Test (Calibré)    : Hit@1={test_metrics_calib.get('hit_at_1')}% | AUC={test_metrics_calib.get('auc')} | Brier={test_metrics_calib.get('brier_score_loss')} | Marge Médiane={test_metrics_calib.get('median_score_gap')}")
    print("=" * 80)

    # Sauvegarde des modèles et métadonnées
    decider_path = output_dir / f"xgb_decider_unified_{timestamp}.json"
    classifier.get_booster().save_model(str(decider_path))

    calibrator_path = None
    if calibrator is not None:
        calibrator_path = output_dir / f"xgb_decider_unified_calibrator_{calibration_method}_{timestamp}.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)

    meta_path = output_dir / f"xgb_decider_unified_meta_{timestamp}.json"
    meta = {
        "timestamp": timestamp,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "fast_ranker_model": str(ranker_model_path),
        "decider_model": str(decider_path),
        "calibrator_model": str(calibrator_path) if calibrator_path else None,
        "calibration_method": calibration_method,
        "classifier_params": CLASSIFIER_PARAMS,
        "metrics": {
            "dev_raw": dev_metrics_raw,
            "dev_calibrated": dev_metrics_calib,
            "test_raw": test_metrics_raw,
            "test_calibrated": test_metrics_calib,
        },
        "splits_count": {
            "train_queries": len(train_groups),
            "dev_queries": len(dev_groups),
            "test_queries": len(test_groups),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModèles et métadonnées sauvegardés :")
    print(f"  Decider    : {decider_path}")
    if calibrator_path:
        print(f"  Calibrator : {calibrator_path}")
    print(f"  Metadata   : {meta_path}\n")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Unified Stage 2 XGBoost Decider")
    parser.add_argument("--crm-path", type=Path, default=PROJECT_ROOT / "data" / "crm_ok_gt_merged_v1.csv")
    parser.add_argument("--partitions-dir", type=Path, default=PROJECT_ROOT / "data" / "candidates_v8_national")
    parser.add_argument("--ranker-model", type=Path, default=PROJECT_ROOT / "models" / "xgbranker_fast_v9.json")
    parser.add_argument("--ban-db", type=Path, default=PROJECT_ROOT / "data" / "ban.duckdb")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of queries to sample")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--chunk-size", type=int, default=250, help="Queries per atomic chunk")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K candidates per query from Stage 1")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable auto-resume")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid", "none"], default="isotonic")
    parser.add_argument("--unattended", action="store_true", default=False, help="Run in unattended mode without pausing on inactivity or screen lock")

    args = parser.parse_args()

    train_unified_decider(
        crm_path=args.crm_path,
        partitions_dir=args.partitions_dir,
        ranker_model_path=args.ranker_model,
        ban_db_path=args.ban_db,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        resume=args.resume,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        calibration_method=args.calibration,
        unattended=args.unattended,
    )


if __name__ == "__main__":
    main()
