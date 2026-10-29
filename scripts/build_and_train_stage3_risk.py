#!/usr/bin/env python3
"""
Pipeline for building Stage 3 scenes from Stage 2 Decider + Tie-Breaker,
then training and calibrating the Stage 3 Routing Risk Model.

Target: >= 99.84% Precision (target_fp_rate <= 0.0016).
Calibration: Isotonic Regression on Dev split.
"""
from __future__ import annotations

import glob
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_matcher.calibration import IsotonicCalibrator, load_calibrator
from src.xgb_matcher.features import FEATURE_NAMES
from src.xgb_matcher.routing_risk import BASE_FEATURES, default_feature_columns
from src.xgb_matcher.tie_breaker import apply_stage2_tie_breakers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_and_train_stage3_risk")


def extract_scene_rows_from_chunks(
    chunk_dir: Path,
    decider_path: Path,
    calibrator_path: Path,
) -> pd.DataFrame:
    """Extract query-level scene features from cached decider training chunks."""
    logger.info("Loading Decider model: %s", decider_path)
    booster = xgb.Booster()
    booster.load_model(str(decider_path))

    logger.info("Loading Decider calibrator: %s", calibrator_path)
    calibrator = load_calibrator(calibrator_path)

    chunk_files = sorted(glob.glob(str(chunk_dir / "*.npz")))
    logger.info("Found %d chunk files in %s", len(chunk_files), chunk_dir)

    feat_indices = {fn: FEATURE_NAMES.index(fn) for fn in BASE_FEATURES if fn in FEATURE_NAMES}
    feature_cols = default_feature_columns(BASE_FEATURES)

    scene_rows = []
    splits = [("train", "train_X", "train_y", "train_groups"),
              ("dev", "dev_X", "dev_y", "dev_groups"),
              ("test", "test_X", "test_y", "test_groups")]

    total_queries = {s[0]: 0 for s in splits}
    hit1_counts = {s[0]: 0 for s in splits}

    for chunk_file in chunk_files:
        data = np.load(chunk_file, allow_pickle=True)
        for split_name, x_key, y_key, g_key in splits:
            X_mat = data[x_key]
            y_vec = data[y_key]
            groups = data[g_key]

            if len(groups) == 0:
                continue

            # Batch score chunk
            dmat = xgb.DMatrix(X_mat, feature_names=FEATURE_NAMES)
            raw_scores = booster.predict(dmat)
            try:
                cal_scores = calibrator.predict_proba(X_mat)[:, 1]
            except Exception:
                cal_scores = raw_scores

            offset = 0
            for g_size in groups:
                q_X = X_mat[offset : offset + g_size]
                q_y = y_vec[offset : offset + g_size]
                q_scores = cal_scores[offset : offset + g_size]
                offset += g_size

                if np.sum(q_y) == 0:
                    continue  # No GT in pool

                total_queries[split_name] += 1
                gt_idx = int(np.argmax(q_y))

                # Build candidate representations for tie-breaker
                cands = [{"siret": f"{i:09d}00010"} for i in range(g_size)]
                new_order, tie_meta = apply_stage2_tie_breakers(
                    retained_cands=cands,
                    s2_scores=q_scores,
                    s2_matrix=q_X,
                    feature_names=FEATURE_NAMES,
                    max_k=5,
                    min_score_floor=0.25,
                    intra_siren_max_gap=0.20,
                    inter_siren_max_gap=0.06,
                )

                top1_idx = new_order[0]
                top2_idx = new_order[1] if len(new_order) > 1 else top1_idx

                is_hit1 = int(top1_idx == gt_idx)
                if is_hit1:
                    hit1_counts[split_name] += 1

                score_top1 = float(q_scores[top1_idx])
                score_top2 = float(q_scores[top2_idx]) if len(new_order) > 1 else 0.0
                score_gap = score_top1 - score_top2
                score_ratio = score_top1 / max(score_top2, 0.001)

                top1_feat = q_X[top1_idx]
                top2_feat = q_X[top2_idx] if len(new_order) > 1 else np.zeros_like(top1_feat)

                row_dict = {
                    "split": split_name,
                    "label_top1_strict": is_hit1,
                    "label_top1_siren": is_hit1,
                    "score_top1": score_top1,
                    "score_top2": score_top2,
                    "score_gap": score_gap,
                    "score_ratio": score_ratio,
                }

                for bf, fi in feat_indices.items():
                    v1 = float(top1_feat[fi])
                    v2 = float(top2_feat[fi])
                    row_dict[f"top1_{bf}"] = v1
                    row_dict[f"top2_{bf}"] = v2
                    row_dict[f"delta_{bf}"] = v1 - v2

                scene_rows.append(row_dict)

    df_scenes = pd.DataFrame(scene_rows)
    for s_name in ["train", "dev", "test"]:
        tot = total_queries[s_name]
        h1 = hit1_counts[s_name]
        rate = (h1 / tot * 100) if tot > 0 else 0.0
        logger.info("Split %s: %d queries | Hit@1: %d (%.2f%%)", s_name, tot, h1, rate)

    return df_scenes


def evaluate_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    target_fp_rate: float = 0.0016,
) -> Tuple[float, List[dict], bool]:
    """Find the threshold respecting target FP rate (target_fp_rate <= 0.0016 for 99.84% precision)."""
    rows = []
    best_threshold = 0.50
    best_auto_rate = -1.0
    target_met = False

    thresholds = np.arange(0.50, 1.00, 0.005)
    for thr in thresholds:
        auto_mask = probs >= thr
        auto_count = int(auto_mask.sum())
        if auto_count == 0:
            continue
        auto_correct = int(y_true[auto_mask].sum())
        fp_count = auto_count - auto_correct
        fp_rate = fp_count / auto_count if auto_count else 0.0
        auto_rate = auto_count / len(y_true) if len(y_true) else 0.0

        rows.append({
            "threshold": float(thr),
            "auto_rate": float(auto_rate),
            "fp_rate": float(fp_rate),
            "precision": float(auto_correct / auto_count if auto_count else 1.0),
            "auto_count": auto_count,
            "fp_count": fp_count,
        })

        if fp_rate <= target_fp_rate and auto_rate > best_auto_rate:
            best_threshold = float(thr)
            best_auto_rate = auto_rate
            target_met = True

    if not target_met and rows:
        # Fallback to minimum FP rate
        fallback = min(rows, key=lambda r: (r["fp_rate"], -r["auto_rate"]))
        best_threshold = fallback["threshold"]
        logger.warning(
            "Target FP rate %.4f not met. Best available threshold=%.3f, FP rate=%.4f (Precision: %.2f%%)",
            target_fp_rate, best_threshold, fallback["fp_rate"], fallback["precision"] * 100,
        )

    return best_threshold, rows, target_met


def main() -> None:
    chunk_dir = PROJECT_ROOT / "data" / "train_checkpoints" / "unified_decider_h1max_10000_seed42"
    decider_path = PROJECT_ROOT / "models" / "xgb_decider_unified_20261025_082618.json"
    calibrator_path = PROJECT_ROOT / "models" / "xgb_decider_unified_calibrator_isotonic_20261025_082618.pkl"

    logger.info("=" * 80)
    logger.info("RECALIBRATION OFFICIELLE DU STAGE 3 RISK MODEL AVEC TIE-BREAKER ENGINE")
    logger.info("=" * 80)

    # 1. Extraction des scènes
    df_scenes = extract_scene_rows_from_chunks(chunk_dir, decider_path, calibrator_path)
    parquet_path = PROJECT_ROOT / "data" / "routing_risk_scenes_v7.parquet"
    df_scenes.to_parquet(parquet_path, index=False)
    logger.info("Saved %d scene records to %s", len(df_scenes), parquet_path)

    # 2. Préparation des jeux d'entraînement
    feature_cols = default_feature_columns(BASE_FEATURES)
    label_col = "label_top1_strict"

    train_df = df_scenes[df_scenes["split"] == "train"].copy()
    dev_df = df_scenes[df_scenes["split"] == "dev"].copy()
    test_df = df_scenes[df_scenes["split"] == "test"].copy()

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)

    X_dev = dev_df[feature_cols].values.astype(np.float32)
    y_dev = dev_df[label_col].values.astype(np.int32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.int32)

    logger.info("Training XGBoost Risk Model on %d train queries...", len(X_train))
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
    logger.info("XGBoost Risk Model trained: %d rounds", clf.best_iteration)

    # 3. Calibration isotonique sur le dev set
    logger.info("Calibrating Stage 3 Risk Model with Isotonic Regression on Dev split (%d queries)...", len(X_dev))
    raw_dev_proba = clf.predict_proba(X_dev)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_dev_proba, y_dev)
    risk_calibrator = IsotonicCalibrator(clf, iso)

    # 4. Évaluation du seuil optimal @ 99.84% précision (cible FP <= 0.0016)
    cal_dev_proba = risk_calibrator.predict_proba(X_dev)[:, 1]
    best_threshold, dev_rows, target_met = evaluate_thresholds(cal_dev_proba, y_dev, target_fp_rate=0.0016)
    logger.info("Optimal threshold on Dev set: %.3f (Target met: %s)", best_threshold, target_met)

    # 5. Validation finale sur le test set (1414 requêtes unseen)
    cal_test_proba = risk_calibrator.predict_proba(X_test)[:, 1]
    test_auto_mask = cal_test_proba >= best_threshold
    test_auto_count = int(test_auto_mask.sum())
    test_auto_correct = int(y_test[test_auto_mask].sum())
    test_fp_count = test_auto_count - test_auto_correct
    test_auto_rate = test_auto_count / len(y_test) * 100
    test_precision = (test_auto_correct / test_auto_count * 100) if test_auto_count > 0 else 100.0

    logger.info("=" * 80)
    logger.info("PERFORMANCE FINALE DU STAGE 3 CALIBRE SUR TEST SET (%d REQUETES)", len(y_test))
    logger.info("AUTO Rate   : %.2f%% (%d / %d)", test_auto_rate, test_auto_count, len(y_test))
    logger.info("Precision   : %.2f%% (%d FP sur %d AUTO)", test_precision, test_fp_count, test_auto_count)
    logger.info("REVIEW Rate : %.2f%% (%d cas orientes vers Places Fallback)", 100 - test_auto_rate, len(y_test) - test_auto_count)
    logger.info("=" * 80)

    # 6. Sauvegarde des modèles et métadonnées
    output_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "routing_risk_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    calibrator_path_out = output_dir / "routing_risk_calibrator.pkl"
    with open(calibrator_path_out, "wb") as f:
        pickle.dump(risk_calibrator, f)

    meta = {
        "target": "strict",
        "model_type": "xgb",
        "features": feature_cols,
        "threshold": best_threshold,
        "calibration": "isotonic",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "calibrator_path": str(calibrator_path_out.relative_to(PROJECT_ROOT)),
    }
    meta_path = output_dir / "routing_risk_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = {
        "target": "strict",
        "model_type": "xgb",
        "target_fp_rate": 0.0016,
        "target_met": target_met,
        "best_threshold": best_threshold,
        "test_precision": float(test_precision),
        "test_auto_rate": float(test_auto_rate),
        "test_auto_count": test_auto_count,
        "test_fp_count": test_fp_count,
        "test_size": len(y_test),
        "dev_size": len(y_dev),
        "train_size": len(y_train),
        "feature_count": len(feature_cols),
    }
    report_path = PROJECT_ROOT / "reports" / "routing_risk_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Persisted model: %s", model_path)
    logger.info("Persisted calibrator: %s", calibrator_path_out)
    logger.info("Persisted metadata: %s", meta_path)
    logger.info("Persisted report: %s", report_path)


if __name__ == "__main__":
    main()
