#!/usr/bin/env python3
"""
evaluate_geo_resolution.py — Comprehensive Geo-Resolution & Retrieval Recall Evaluator.

Evaluates candidate pool coverage and retrieval recall across a CRM Ground Truth dataset
(e.g., the 17,054 baseline or the 15,516 increment or the 32,570 merged dataset).

Key Safety Features:
  - Checkpoint & Auto-Resume: Automatically resumes interrupted jobs from where they stopped.
  - PowerGuard Monitor: Detects Windows sleep / lid-close / lock events, completes the active query,
    atomically saves the checkpoint, flushes CSV logs, and yields CPU load to 0% to allow safe,
    cool hardware sleep. Auto-resumes when the system wakes up.

Outputs:
  - Console summary table with overall and segment-by-segment recall
  - JSON summary report (reports/geo_validation/geo_benchmark_<name>_summary.json)
  - Detailed CSV log with per-query diagnostic (reports/geo_validation/geo_benchmark_<name>_details.csv)
  - Atomic checkpoints in (reports/geo_validation/.checkpoints/checkpoint_<name>.json)

Usage:
    python scripts/evaluate_geo_resolution.py --crm data/crm_ok_gt.csv --tag 17k_baseline
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.checkpoint import CheckpointManager
from src.xgb_matcher.features import preprocess_crm_row
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.power_guard import PowerMonitor
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache
from src.xgb_matcher.timing import PipelineTimer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_geo_resolution")


# Target SSOT baselines for comparison
BASELINE_TARGETS = {
    "recall_base_min": 99.0,      # Target >= 99.0%
    "recall_prefilter_min": 98.5, # Target >= 98.5%
}


def load_crm_gt(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
    return rows


def evaluate(
    rows: List[Dict[str, str]],
    partitions_dir: Path,
    output_dir: Path,
    tag: str,
    prefilter_k: int = 500,
    siren_to_geo_path: Path | None = None,
    log_file: Path | None = None,
    crm_path: Path | None = None,
    resume: bool = True,
    fresh: bool = False,
    checkpoint_interval: int = 100,
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

    store = PartitionedCandidateStore(partitions_dir)
    config = RetrievalConfigV1(
        prefilter_k=prefilter_k,
        dense_retrieval_enabled=False,
    )

    siren_to_geo = None
    if siren_to_geo_path and siren_to_geo_path.exists():
        try:
            from src.xgb_matcher.siren_retrieval import SirenToGeoIndex
            siren_to_geo = SirenToGeoIndex(siren_to_geo_path)
            logger.info("Loaded SIREN-to-geo mapping from %s", siren_to_geo_path)
        except Exception as e:
            logger.warning("Could not load SIREN-to-geo mapping: %s", e)

    cache = TfidfPersistentCache(config.signature().hash)
    timer = PipelineTimer()

    # Checkpoint management
    ckpt = CheckpointManager(
        tag=tag,
        checkpoint_dir=ckpt_dir,
        total_rows=len(rows),
        crm_path=crm_path,
        config_signature=config.signature().hash,
        save_interval=checkpoint_interval,
    )

    is_resumed = False
    if resume and not fresh and ckpt.is_resumable():
        is_resumed = ckpt.load()

    # Setup Details CSV writer
    details_path = output_dir / f"geo_benchmark_{tag}_details.csv"
    detail_fieldnames = [
        "crm_id",
        "crm_name",
        "crm_insee",
        "crm_cp",
        "gt_siret",
        "loc_type",
        "sirene_etat",
        "base_pool_size",
        "filtered_pool_size",
        "prefilter_pool_size",
        "gt_in_base",
        "gt_in_filtered",
        "gt_in_prefilter",
        "loss_reason",
    ]
    ckpt.init_details_writer(details_path, detail_fieldnames, resume=is_resumed)

    # Power Monitor setup
    power_mon = PowerMonitor(
        pause_on_lock=pause_on_lock,
        idle_timeout_seconds=idle_timeout_seconds,
    )
    if enable_power_monitor:
        power_mon.start()

    logger.info(
        "Evaluating %d CRM rows (Resumed: %s, Starting at idx: %d)...",
        len(rows),
        is_resumed,
        ckpt.last_processed_idx + 1,
    )

    try:
        t_batch_start = time.perf_counter()
        tfidf_cache: OrderedDict = OrderedDict()

        for idx, r in enumerate(rows):
            # Skip rows already processed in previous run
            if is_resumed and idx <= ckpt.last_processed_idx:
                continue

            gt_siret = (r.get("gt_siret") or r.get("ground_truth_siret") or "").strip()
            if not gt_siret:
                continue

            t_query_0 = time.perf_counter()

            crm_insee = (r.get("crm_insee") or r.get("insee") or "").strip()
            crm_cp = (r.get("crm_cp") or r.get("postcode") or "").strip()
            crm_name = (r.get("crm_name") or r.get("nom") or "").strip()
            crm_adresse = (r.get("crm_adresse") or r.get("adresse") or "").strip()
            crm_commune = (r.get("crm_commune") or r.get("ville") or "").strip()
            crm_id = (r.get("crm_id") or r.get("id") or f"ROW_{idx}").strip()
            loc_type = (r.get("loc_match_type") or ("INSEE" if crm_insee else "CP_ONLY")).strip()
            etat = (r.get("sirene_etat") or "UNKNOWN").strip().upper()

            crm_row = {
                "crm_id": crm_id,
                "crm_name": crm_name,
                "crm_address": crm_adresse,
                "crm_city": crm_commune,
                "postcode": crm_cp,
                "insee": crm_insee,
            }
            crm_pre = preprocess_crm_row(crm_row)

            # Cap in-memory TF-IDF cache to 300 entries max to prevent RAM saturation (LRU)
            while len(tfidf_cache) > 300:
                tfidf_cache.popitem(last=False)

            try:
                res = build_candidate_pool(
                    store=store,
                    crm_row=crm_row,
                    crm_pre=crm_pre,
                    config=config,
                    tfidf_cache=tfidf_cache,
                    gt_siret=gt_siret,
                    persistent_cache=cache,
                    timer=timer,
                    siren_to_geo=siren_to_geo,
                )

                gt_in_base = bool(res.gt_in_base_pool)
                gt_in_filtered = bool(res.gt_in_filtered_pool)
                gt_in_prefilter = bool(res.gt_in_tfidf_pool)
                loss_reason = res.loss_reason or ("NOT_IN_BASE" if not gt_in_base else "")
                base_pool_sz = res.pool_sizes.get("base", 0)
                filtered_pool_sz = res.pool_sizes.get("filtered", 0)
                prefilter_pool_sz = len(res.candidates)
            except Exception as e:
                logger.error(
                    "QUERY_ERROR at row %d (crm_id=%s, insee=%s, cp=%s): %s",
                    idx, crm_id, crm_insee, crm_cp, e, exc_info=True,
                )
                gt_in_base = False
                gt_in_filtered = False
                gt_in_prefilter = False
                loss_reason = "QUERY_ERROR"
                base_pool_sz = 0
                filtered_pool_sz = 0
                prefilter_pool_sz = 0
                gc.collect()

            detail_entry = {
                "crm_id": crm_id,
                "crm_name": crm_name,
                "crm_insee": crm_insee,
                "crm_cp": crm_cp,
                "gt_siret": gt_siret,
                "loc_type": loc_type,
                "sirene_etat": etat,
                "base_pool_size": base_pool_sz,
                "filtered_pool_size": filtered_pool_sz,
                "prefilter_pool_size": prefilter_pool_sz,
                "gt_in_base": gt_in_base,
                "gt_in_filtered": gt_in_filtered,
                "gt_in_prefilter": gt_in_prefilter,
                "loss_reason": loss_reason,
            }

            delta_t = time.perf_counter() - t_query_0
            ckpt.record_query(
                idx=idx,
                gt_in_base=gt_in_base,
                gt_in_filtered=gt_in_filtered,
                gt_in_prefilter=gt_in_prefilter,
                loc_type=loc_type,
                etat=etat,
                loss_reason=loss_reason,
                detail_row=detail_entry,
                elapsed_delta=delta_t,
            )

            # Periodic garbage collection at checkpoint boundaries
            if ckpt.total_queries % checkpoint_interval == 0:
                gc.collect()

            if ckpt.total_queries % 2000 == 0:
                logger.info(
                    "Processed %d/%d (%.1f%%) — Base Recall: %.2f%%, Prefilter@%d: %.2f%%",
                    ckpt.total_queries,
                    len(rows),
                    (ckpt.total_queries / len(rows)) * 100,
                    (ckpt.gt_in_base_count / max(ckpt.total_queries, 1)) * 100,
                    prefilter_k,
                    (ckpt.gt_in_prefilter_count / max(ckpt.total_queries, 1)) * 100,
                )

            # Check if power monitor signaled sleep / lid close / lock
            if enable_power_monitor and power_mon.should_pause():
                logger.info("[CHECKPOINT] Flushing state before entering safe pause...")
                gc.collect()
                ckpt.save(status="PAUSED", force=True)
                power_mon.wait_for_resume()
                logger.info("[CHECKPOINT] Resuming loop after safe sleep pause.")

    finally:
        if enable_power_monitor:
            power_mon.stop()

    total = ckpt.total_queries
    elapsed = ckpt.elapsed_seconds
    qps = total / max(elapsed, 0.001)

    gt_in_base_count = ckpt.gt_in_base_count
    gt_in_filtered_count = ckpt.gt_in_filtered_count
    gt_in_prefilter_count = ckpt.gt_in_prefilter_count

    recall_base = (gt_in_base_count / total * 100) if total else 0.0
    recall_filtered = (gt_in_filtered_count / total * 100) if total else 0.0
    recall_prefilter = (gt_in_prefilter_count / total * 100) if total else 0.0
    recall_prefilter_given_base = (
        (gt_in_prefilter_count / gt_in_base_count * 100) if gt_in_base_count else 0.0
    )

    summary = {
        "tag": tag,
        "total_queries": total,
        "elapsed_seconds": round(elapsed, 2),
        "queries_per_second": round(qps, 1),
        "recall_base_pct": round(recall_base, 2),
        "recall_filtered_pct": round(recall_filtered, 2),
        "recall_prefilter_pct": round(recall_prefilter, 2),
        "recall_prefilter_given_base_pct": round(recall_prefilter_given_base, 2),
        "gt_in_base_count": gt_in_base_count,
        "gt_in_prefilter_count": gt_in_prefilter_count,
        "by_loc_match_type": {
            k: {
                "total": v["total"],
                "base_recall_pct": round((v["base"] / v["total"] * 100), 2) if v["total"] else 0,
                "prefilter_recall_pct": round((v["prefilter"] / v["total"] * 100), 2) if v["total"] else 0,
            }
            for k, v in ckpt.by_loc_type.items()
        },
        "by_sirene_etat": {
            k: {
                "total": v["total"],
                "base_recall_pct": round((v["base"] / v["total"] * 100), 2) if v["total"] else 0,
                "prefilter_recall_pct": round((v["prefilter"] / v["total"] * 100), 2) if v["total"] else 0,
            }
            for k, v in ckpt.by_etat.items()
        },
        "loss_reasons": dict(ckpt.loss_reasons.most_common()),
    }

    # Save summary JSON
    summary_path = output_dir / f"geo_benchmark_{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary report to %s", summary_path)
    logger.info("Saved query-level details to %s", details_path)

    # Finalize checkpoint (archives checkpoint so future fresh runs execute from 0)
    ckpt.finalize_completed(summary)

    # Print nicely formatted console report
    print("\n" + "=" * 78)
    print(f"[REPORT] GEO-RESOLUTION & RETRIEVAL RECALL — {tag.upper()}")
    print("=" * 78)
    print(f"  Total Queries Evaluated:       {total:>7,}")
    print(f"  Execution Time:                {elapsed:>7.1f}s ({qps:>5.1f} queries/s)")
    print(f"  GT in Base Partition (Pool):   {gt_in_base_count:>7,} / {total:,} ({recall_base:.2f}%)")
    print(f"  GT in Filtered Pool:           {gt_in_filtered_count:>7,} / {total:,} ({recall_filtered:.2f}%)")
    print(f"  GT in Top-{prefilter_k} Prefilter:      {gt_in_prefilter_count:>7,} / {total:,} ({recall_prefilter:.2f}%)")
    print(f"  Prefilter Recall (given Base): {recall_prefilter_given_base:.2f}%")
    print("-" * 78)
    print("Location Match Type Breakdown:")
    for loc_k, loc_v in summary["by_loc_match_type"].items():
        print(
            f"   - {loc_k:<30} Count: {loc_v['total']:>6,} | Base: {loc_v['base_recall_pct']:>5.2f}% | Top-{prefilter_k}: {loc_v['prefilter_recall_pct']:>5.2f}%"
        )
    print("-" * 78)
    print("Administrative State Breakdown (Actif / Ferme):")
    for etat_k, etat_v in summary["by_sirene_etat"].items():
        print(
            f"   - {etat_k:<30} Count: {etat_v['total']:>6,} | Base: {etat_v['base_recall_pct']:>5.2f}% | Top-{prefilter_k}: {etat_v['prefilter_recall_pct']:>5.2f}%"
        )
    print("-" * 78)
    print("Loss Reasons Breakdown:")
    for r_k, r_v in ckpt.loss_reasons.most_common():
        print(f"   - {r_k:<30} {r_v:>6,} cases ({(r_v / max(total, 1) * 100):.2f}%)")
    print("=" * 78)

    # Regression check
    regression = False
    if recall_base < BASELINE_TARGETS["recall_base_min"]:
        print(f"[WARNING] Base recall {recall_base:.2f}% is below target {BASELINE_TARGETS['recall_base_min']}%.")
        regression = True
    if recall_prefilter < BASELINE_TARGETS["recall_prefilter_min"]:
        print(
            f"[WARNING] Prefilter recall {recall_prefilter:.2f}% is below target {BASELINE_TARGETS['recall_prefilter_min']}%."
        )
        regression = True
    if not regression:
        print("[OK] QUALITY VERDICT: NO REGRESSION DETECTED. ALL TARGETS MET.")
    print("=" * 78 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Geo-Resolution & Retrieval Recall")
    parser.add_argument("--crm", type=Path, default=Path("data/crm_ok_gt.csv"), help="CRM Ground Truth CSV")
    parser.add_argument(
        "--partitions-dir", type=Path, default=Path("data/candidates_v7_all"), help="Partitions directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/geo_validation"), help="Output reports directory"
    )
    parser.add_argument("--tag", type=str, default="17k_baseline", help="Tag for report filenames")
    parser.add_argument("--prefilter-k", type=int, default=500, help="Prefilter Top-K candidates")
    parser.add_argument(
        "--siren-to-geo",
        type=Path,
        default=Path("data/siren_index/siren_to_geo.parquet"),
        help="SIREN-to-geo index path",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 for all)")
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="Disable auto-resume from checkpoint"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Force fresh restart (ignore existing checkpoint)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100, help="Save checkpoint every N queries"
    )
    parser.add_argument(
        "--no-power-monitor",
        dest="enable_power_monitor",
        action="store_false",
        help="Disable Windows power / sleep monitor",
    )
    parser.add_argument(
        "--pause-on-lock",
        dest="pause_on_lock",
        action="store_true",
        default=True,
        help="Pause computation when Windows session is locked (Default: True)",
    )
    parser.add_argument(
        "--no-pause-on-lock",
        dest="pause_on_lock",
        action="store_false",
        help="Disable automatic pausing on Windows session lock",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="Inactivity timeout in seconds before pausing computation (Default: 300.0 = 5 min, 0 to disable)",
    )

    args = parser.parse_args()

    if not args.crm.exists():
        logger.error("CRM file not found: %s", args.crm)
        sys.exit(1)

    rows = load_crm_gt(args.crm)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]

    siren_geo = args.siren_to_geo if args.siren_to_geo.exists() else None
    evaluate(
        rows=rows,
        partitions_dir=args.partitions_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        prefilter_k=args.prefilter_k,
        siren_to_geo_path=siren_geo,
        log_file=args.log_file,
        crm_path=args.crm,
        resume=args.resume,
        fresh=args.fresh,
        checkpoint_interval=args.checkpoint_interval,
        enable_power_monitor=args.enable_power_monitor,
        pause_on_lock=args.pause_on_lock,
        idle_timeout_seconds=args.idle_timeout,
    )


if __name__ == "__main__":
    main()
