#!/usr/bin/env python3
"""
evaluate_geo_resolution.py — Comprehensive Geo-Resolution & Retrieval Recall Evaluator.

Evaluates candidate pool coverage and retrieval recall across a CRM Ground Truth dataset
(e.g., the 17,054 baseline or the 15,516 increment or the 32,570 merged dataset).

Outputs:
  - Console summary table with overall and segment-by-segment recall
  - JSON summary report (reports/geo_validation/summary_<name>.json)
  - Detailed CSV log with per-query diagnostic (reports/geo_validation/details_<name>.csv)

Usage:
    python scripts/evaluate_geo_resolution.py --crm data/crm_ok_gt.csv --tag 17k_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.partitioned_store import PartitionedCandidateStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.features import preprocess_crm_row
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
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

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

    total = 0
    gt_in_base_count = 0
    gt_in_filtered_count = 0
    gt_in_prefilter_count = 0

    # Stratified stats
    by_loc_type: Dict[str, Dict[str, int]] = {}
    by_etat: Dict[str, Dict[str, int]] = {}
    loss_reasons: Counter = Counter()

    details: List[Dict[str, Any]] = []

    logger.info("Evaluating %d CRM ground truth rows...", len(rows))
    t0 = time.perf_counter()

    for idx, r in enumerate(rows):
        gt_siret = (r.get("gt_siret") or r.get("ground_truth_siret") or "").strip()
        if not gt_siret:
            continue

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

        tfidf_cache: Dict = {}
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

        total += 1
        gt_in_base = bool(res.gt_in_base_pool)
        gt_in_filtered = bool(res.gt_in_filtered_pool)
        gt_in_prefilter = bool(res.gt_in_tfidf_pool)
        loss_reason = res.loss_reason or ("NOT_IN_BASE" if not gt_in_base else "")

        if gt_in_base:
            gt_in_base_count += 1
        if gt_in_filtered:
            gt_in_filtered_count += 1
        if gt_in_prefilter:
            gt_in_prefilter_count += 1
        if loss_reason:
            loss_reasons[loss_reason] += 1

        # Stratify by loc_type
        if loc_type not in by_loc_type:
            by_loc_type[loc_type] = {"total": 0, "base": 0, "prefilter": 0}
        by_loc_type[loc_type]["total"] += 1
        if gt_in_base:
            by_loc_type[loc_type]["base"] += 1
        if gt_in_prefilter:
            by_loc_type[loc_type]["prefilter"] += 1

        # Stratify by etat
        if etat not in by_etat:
            by_etat[etat] = {"total": 0, "base": 0, "prefilter": 0}
        by_etat[etat]["total"] += 1
        if gt_in_base:
            by_etat[etat]["base"] += 1
        if gt_in_prefilter:
            by_etat[etat]["prefilter"] += 1

        details.append({
            "crm_id": crm_id,
            "crm_name": crm_name,
            "crm_insee": crm_insee,
            "crm_cp": crm_cp,
            "gt_siret": gt_siret,
            "loc_type": loc_type,
            "sirene_etat": etat,
            "base_pool_size": res.pool_sizes.get("base", 0),
            "filtered_pool_size": res.pool_sizes.get("filtered", 0),
            "prefilter_pool_size": len(res.candidates),
            "gt_in_base": gt_in_base,
            "gt_in_filtered": gt_in_filtered,
            "gt_in_prefilter": gt_in_prefilter,
            "loss_reason": loss_reason,
        })

        if total % 2000 == 0:
            logger.info(
                "Processed %d/%d (%.1f%%) — Base Recall: %.2f%%, Prefilter@%d: %.2f%%",
                total, len(rows), (total / len(rows)) * 100,
                (gt_in_base_count / total) * 100,
                prefilter_k,
                (gt_in_prefilter_count / total) * 100,
            )

    elapsed = time.perf_counter() - t0
    qps = total / max(elapsed, 0.001)

    recall_base = (gt_in_base_count / total * 100) if total else 0.0
    recall_filtered = (gt_in_filtered_count / total * 100) if total else 0.0
    recall_prefilter = (gt_in_prefilter_count / total * 100) if total else 0.0
    recall_prefilter_given_base = (gt_in_prefilter_count / gt_in_base_count * 100) if gt_in_base_count else 0.0

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
            for k, v in by_loc_type.items()
        },
        "by_sirene_etat": {
            k: {
                "total": v["total"],
                "base_recall_pct": round((v["base"] / v["total"] * 100), 2) if v["total"] else 0,
                "prefilter_recall_pct": round((v["prefilter"] / v["total"] * 100), 2) if v["total"] else 0,
            }
            for k, v in by_etat.items()
        },
        "loss_reasons": dict(loss_reasons.most_common()),
    }

    # Save summary JSON
    summary_path = output_dir / f"geo_benchmark_{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary report to %s", summary_path)

    # Save details CSV
    details_path = output_dir / f"geo_benchmark_{tag}_details.csv"
    with open(details_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(details)
    logger.info("Saved query-level details to %s", details_path)

    # Print nicely formatted console report
    print("\n" + "=" * 78)
    print(f"📊 GEO-RESOLUTION & RETRIEVAL RECALL REPORT — {tag.upper()}")
    print("=" * 78)
    print(f"  Total Queries Evaluated:       {total:>7,}")
    print(f"  Execution Time:                {elapsed:>7.1f}s ({qps:>5.1f} queries/s)")
    print(f"  GT in Base Partition (Pool):   {gt_in_base_count:>7,} / {total:,} ({recall_base:.2f}%)")
    print(f"  GT in Filtered Pool:           {gt_in_filtered_count:>7,} / {total:,} ({recall_filtered:.2f}%)")
    print(f"  GT in Top-{prefilter_k} Prefilter:      {gt_in_prefilter_count:>7,} / {total:,} ({recall_prefilter:.2f}%)")
    print(f"  Prefilter Recall (given Base): {recall_prefilter_given_base:.2f}%")
    print("-" * 78)
    print("📍 Breakdown by Location Match Type:")
    for loc_k, loc_v in summary["by_loc_match_type"].items():
        print(f"   • {loc_k:<30} Count: {loc_v['total']:>6,} | Base: {loc_v['base_recall_pct']:>5.2f}% | Top-{prefilter_k}: {loc_v['prefilter_recall_pct']:>5.2f}%")
    print("-" * 78)
    print("🏢 Breakdown by Administrative State (Actif / Ferme):")
    for etat_k, etat_v in summary["by_sirene_etat"].items():
        print(f"   • {etat_k:<30} Count: {etat_v['total']:>6,} | Base: {etat_v['base_recall_pct']:>5.2f}% | Top-{prefilter_k}: {etat_v['prefilter_recall_pct']:>5.2f}%")
    print("-" * 78)
    print("⚠️ Loss Reasons Breakdown:")
    for r_k, r_v in loss_reasons.most_common():
        print(f"   • {r_k:<30} {r_v:>6,} cases ({(r_v / total * 100):.2f}%)")
    print("=" * 78)

    # Regression check
    regression = False
    if recall_base < BASELINE_TARGETS["recall_base_min"]:
        print(f"❌ REGRESSION WARNING: Base recall {recall_base:.2f}% is below target {BASELINE_TARGETS['recall_base_min']}%.")
        regression = True
    if recall_prefilter < BASELINE_TARGETS["recall_prefilter_min"]:
        print(f"❌ REGRESSION WARNING: Prefilter recall {recall_prefilter:.2f}% is below target {BASELINE_TARGETS['recall_prefilter_min']}%.")
        regression = True
    if not regression:
        print("✅ QUALITY VERDICT: NO REGRESSION DETECTED. ALL TARGETS MET.")
    print("=" * 78 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Geo-Resolution & Retrieval Recall")
    parser.add_argument("--crm", type=Path, default=Path("data/crm_ok_gt.csv"), help="CRM Ground Truth CSV")
    parser.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"), help="Partitions directory")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/geo_validation"), help="Output reports directory")
    parser.add_argument("--tag", type=str, default="17k_baseline", help="Tag for report filenames")
    parser.add_argument("--prefilter-k", type=int, default=500, help="Prefilter Top-K candidates")
    parser.add_argument("--siren-to-geo", type=Path, default=Path("data/siren_index/siren_to_geo.parquet"), help="SIREN-to-geo index path")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 for all)")

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
    )


if __name__ == "__main__":
    main()
