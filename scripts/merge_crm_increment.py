#!/usr/bin/env python3
"""
merge_crm_increment.py — Fusion controlee du dataset CRM existant avec le nouvel increment.

Regles d'inclusion dans la fusion :
  - already_in_crm_ok_gt_exact == 'False'   (pas deja present en exact)
  - existing_component_relation == 'UNSEEN_SIREN_NEEDS_ASSIGNMENT'  (SIRENs vraiment nouveaux)

Sortie : data/crm_ok_gt_merged_v1.csv

Usage:
    python scripts/merge_crm_increment.py [--dry-run]
"""

import argparse
import csv
import sys
from pathlib import Path


# Colonnes canoniques du dataset de reference (crm_ok_gt.csv)
CANONICAL_COLUMNS = [
    "crm_name",
    "crm_cp",
    "crm_insee",
    "crm_id",
    "crm_commune",
    "gt_siret",
    "crm_adresse",
    "sirene_insee",
    "sirene_cp",
    "sirene_etat",
    "loc_match_type",
]


def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def save_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CRM increment into main dataset")
    parser.add_argument(
        "--orig",
        type=Path,
        default=Path("data/crm_ok_gt.csv"),
        help="Original dataset (default: data/crm_ok_gt.csv)",
    )
    parser.add_argument(
        "--increment",
        type=Path,
        default=Path("data/crm_ok_gt_increment_20260817.csv"),
        help="Increment CSV (default: data/crm_ok_gt_increment_20260817.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/crm_ok_gt_merged_v1.csv"),
        help="Output merged CSV (default: data/crm_ok_gt_merged_v1.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing the output file",
    )
    args = parser.parse_args()

    # --- Load original dataset ---
    print(f"[1/6] Loading original dataset: {args.orig}")
    if not args.orig.exists():
        print(f"ERROR: {args.orig} not found.", file=sys.stderr)
        sys.exit(1)
    orig_rows = load_csv(args.orig)
    print(f"      Loaded {len(orig_rows):,} rows.")

    # --- Load increment ---
    print(f"[2/6] Loading increment: {args.increment}")
    if not args.increment.exists():
        print(f"ERROR: {args.increment} not found.", file=sys.stderr)
        sys.exit(1)
    inc_rows = load_csv(args.increment)
    print(f"      Loaded {len(inc_rows):,} rows.")

    # --- Filter increment: only UNSEEN, not already present ---
    print("[3/6] Filtering increment rows...")
    new_rows = [
        r for r in inc_rows
        if r.get("already_in_crm_ok_gt_exact") == "False"
        and r.get("existing_component_relation") == "UNSEEN_SIREN_NEEDS_ASSIGNMENT"
    ]
    print(f"      Rows passing filter (UNSEEN + not exact): {len(new_rows):,}")
    print(f"      Rows excluded (already present / quarantine / train): {len(inc_rows) - len(new_rows):,}")

    # --- Fingerprint deduplication (safety check) ---
    print("[4/6] Deduplication check on crm_gt_fingerprint...")
    orig_fingerprints = set(r.get("crm_gt_fingerprint", "") for r in orig_rows)
    before_dedup = len(new_rows)
    new_rows_deduped = [
        r for r in new_rows
        if r.get("crm_gt_fingerprint", "") not in orig_fingerprints
    ]
    dupes_found = before_dedup - len(new_rows_deduped)
    if dupes_found > 0:
        print(f"      WARNING: {dupes_found} rows removed by fingerprint dedup (safety catch).")
    else:
        print(f"      OK — 0 fingerprint duplicates found.")

    # Also deduplicate within the new rows themselves
    seen_fp: set = set()
    new_rows_final = []
    for r in new_rows_deduped:
        fp = r.get("crm_gt_fingerprint", "")
        if fp and fp in seen_fp:
            continue
        seen_fp.add(fp)
        new_rows_final.append(r)
    intra_dupes = len(new_rows_deduped) - len(new_rows_final)
    if intra_dupes > 0:
        print(f"      WARNING: {intra_dupes} intra-increment fingerprint duplicates removed.")

    # --- Statistics ---
    print()
    print("=" * 60)
    print("MERGE STATISTICS")
    print("=" * 60)
    print(f"  Original dataset:              {len(orig_rows):>7,}")
    print(f"  Increment UNSEEN rows:         {len(new_rows_final):>7,}")
    print(f"  Total post-fusion:             {len(orig_rows) + len(new_rows_final):>7,}")
    print()

    # State breakdown in new rows
    actif = sum(1 for r in new_rows_final if r.get("sirene_etat") == "A")
    ferme = sum(1 for r in new_rows_final if r.get("sirene_etat") == "F")
    print(f"  New rows — Actif (A):          {actif:>7,}")
    print(f"  New rows — Ferme (F):          {ferme:>7,}")

    # Loc match type
    from collections import Counter
    loc_counts = Counter(r.get("loc_match_type", "") for r in new_rows_final)
    print()
    print("  loc_match_type distribution (new rows):")
    for k, v in loc_counts.most_common():
        print(f"    {k:<40} {v:>6,}")

    if args.dry_run:
        print()
        print("[DRY RUN] No file written.")
        return

    # --- Merge and write ---
    print()
    print(f"[5/6] Merging rows and writing to: {args.output}")
    merged_rows = orig_rows + new_rows_final
    save_csv(merged_rows, args.output, CANONICAL_COLUMNS)

    # --- Verification ---
    print(f"[6/6] Verification: reading back {args.output}...")
    verify = load_csv(args.output)
    assert len(verify) == len(orig_rows) + len(new_rows_final), (
        f"Row count mismatch: expected {len(orig_rows) + len(new_rows_final)}, got {len(verify)}"
    )
    print(f"      OK — {len(verify):,} rows verified.")
    print()
    print(f"Done. Merged dataset written to: {args.output}")


if __name__ == "__main__":
    main()
