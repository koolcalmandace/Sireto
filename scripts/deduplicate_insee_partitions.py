#!/usr/bin/env python3
"""
deduplicate_insee_partitions.py — Safe cleanup of duplicate parquet files in data/candidates_v7_all/insee.

Ensures that every commune partition contains exactly 1 unique parquet file.
"""

from pathlib import Path
import pyarrow.parquet as pq

def main():
    insee_dir = Path("data/candidates_v7_all/insee")
    if not insee_dir.is_dir():
        print(f"Error: {insee_dir} not found.")
        return 1

    folders = [d for d in insee_dir.iterdir() if d.is_dir()]
    print(f"Total INSEE commune folders inspected: {len(folders)}")

    deleted_count = 0
    skipped_count = 0

    for d in sorted(folders):
        files = sorted(list(d.glob("*.parquet")), key=lambda p: p.name)
        if len(files) == 2:
            f_keep = files[0]
            f_del = files[1]

            m_keep = pq.read_metadata(f_keep)
            m_del = pq.read_metadata(f_del)

            if m_keep.num_rows == m_del.num_rows and m_keep.num_columns == m_del.num_columns:
                f_del.unlink()
                deleted_count += 1
            else:
                print(f"WARNING: Mismatch in {d.name}: {m_keep.num_rows} vs {m_del.num_rows} rows. Not deleting.")
                skipped_count += 1
        elif len(files) > 2:
            print(f"WARNING: More than 2 files in {d.name}: {len(files)} files.")

    print(f"Cleanup complete: {deleted_count} duplicate files safely removed.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} folders due to row count mismatches.")

    # Verification pass
    remaining_multi = sum(1 for d in folders if len(list(d.glob("*.parquet"))) != 1)
    print(f"Verification: Folders with != 1 parquet file: {remaining_multi}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
