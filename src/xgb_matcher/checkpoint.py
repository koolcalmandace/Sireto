"""
checkpoint.py — Robust per-query JSON checkpointing and auto-resume management.

Ensures long-running batch and evaluation jobs survive unexpected shutdowns,
crashes, and power events by:
  1. Persisting execution state atomically (write-to-temp + atomic replace).
  2. Incrementally streaming query detail logs to disk.
  3. Seamlessly resuming from the exact query where execution stopped.
  4. Finalizing and cleaning active checkpoints on completion so future runs start fresh.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("xgb_matcher.checkpoint")


class CheckpointManager:
    """Manages atomic state persistence and resume capabilities for batch jobs."""

    def __init__(
        self,
        tag: str,
        checkpoint_dir: Path,
        total_rows: int,
        crm_path: Optional[Path] = None,
        config_signature: Optional[str] = None,
        save_interval: int = 100,
    ) -> None:
        self.tag = tag
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{tag}.json"
        self.backup_file = self.checkpoint_dir / f"checkpoint_{tag}.bak"
        self.completed_marker = self.checkpoint_dir / f"checkpoint_{tag}.completed.json"

        self.total_rows = total_rows
        self.crm_path_str = str(crm_path.resolve()) if crm_path else ""
        self.config_signature = config_signature or ""
        self.save_interval = max(1, save_interval)

        # In-memory state
        self.last_processed_idx = -1
        self.total_queries = 0
        self.gt_in_base_count = 0
        self.gt_in_filtered_count = 0
        self.gt_in_prefilter_count = 0
        self.by_loc_type: Dict[str, Dict[str, int]] = {}
        self.by_etat: Dict[str, Dict[str, int]] = {}
        self.loss_reasons: Counter = Counter()
        self.elapsed_seconds: float = 0.0
        self.status: str = "INITIALIZED"  # INITIALIZED, IN_PROGRESS, PAUSED, COMPLETED
        self._unsaved_updates = 0
        self._details_file_handle = None
        self._details_csv_writer = None
        self._details_path: Optional[Path] = None

    def is_resumable(self) -> bool:
        """Check if a valid, uncompleted checkpoint exists on disk."""
        if not self.checkpoint_file.exists():
            return False
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "COMPLETED":
                return False
            # Basic sanity checks on dataset
            if self.crm_path_str and data.get("crm_path") and data.get("crm_path") != self.crm_path_str:
                logger.warning(
                    "Checkpoint exists but dataset path differs (%s vs %s)",
                    data.get("crm_path"), self.crm_path_str
                )
                return False
            return data.get("last_processed_idx", -1) >= 0
        except Exception as e:
            logger.warning("Failed to inspect checkpoint file %s: %s", self.checkpoint_file, e)
            return False

    def load(self) -> bool:
        """Load state from existing checkpoint."""
        if not self.checkpoint_file.exists():
            return False

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Corrupted checkpoint %s (%s). Trying backup...", self.checkpoint_file, e)
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as be:
                    logger.error("Backup checkpoint also corrupted: %s", be)
                    return False
            else:
                return False

        self.last_processed_idx = data.get("last_processed_idx", -1)
        self.total_queries = data.get("total_queries", 0)
        self.gt_in_base_count = data.get("gt_in_base_count", 0)
        self.gt_in_filtered_count = data.get("gt_in_filtered_count", 0)
        self.gt_in_prefilter_count = data.get("gt_in_prefilter_count", 0)
        self.by_loc_type = data.get("by_loc_type", {})
        self.by_etat = data.get("by_etat", {})
        self.loss_reasons = Counter(data.get("loss_reasons", {}))
        self.elapsed_seconds = data.get("elapsed_seconds", 0.0)
        self.status = "IN_PROGRESS"

        logger.info(
            "Resumed checkpoint '%s': already processed %d queries (up to row index %d)",
            self.tag, self.total_queries, self.last_processed_idx
        )
        return True

    def get_processed_crm_ids(self, details_path: Optional[Path] = None) -> Set[str]:
        """Return the set of unique crm_ids already present in the details CSV."""
        path = Path(details_path) if details_path else self._details_path
        if not path or not path.exists():
            return set()
        processed: Set[str] = set()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=";")
                header = next(reader, None)
                if not header:
                    return set()
                crm_id_col = 0
                for i, col in enumerate(header):
                    if col.strip().lower() in ("crm_id", "id"):
                        crm_id_col = i
                        break
                for row in reader:
                    if row and len(row) > crm_id_col:
                        cid = row[crm_id_col].strip()
                        if cid:
                            processed.add(cid)
        except Exception as e:
            logger.warning("Error reading processed IDs from %s: %s", path, e)
        return processed

    @staticmethod
    def _count_file_lines(path: Path) -> int:
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
        return count

    def sync_from_details_csv(self, details_path: Path, fieldnames: List[str]) -> Set[str]:
        """
        Deduplicates details CSV if needed, rebuilds state metrics (total_queries, gt counts,
        stratification) from ground-truth details rows, and returns the set of completed crm_ids.
        """
        path = Path(details_path)
        if not path.exists() or path.stat().st_size == 0:
            return set()

        seen_ids: Set[str] = set()
        unique_rows: List[Dict[str, Any]] = []

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    cid = (row.get("crm_id") or row.get("id") or "").strip()
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        unique_rows.append(row)
        except Exception as e:
            logger.warning("Error reading details CSV for sync: %s", e)
            return self.get_processed_crm_ids(path)

        # Reconstruct metrics
        self.total_queries = len(unique_rows)
        self.gt_in_base_count = 0
        self.gt_in_filtered_count = 0
        self.gt_in_prefilter_count = 0
        self.by_loc_type.clear()
        self.by_etat.clear()
        self.loss_reasons.clear()

        for r in unique_rows:
            gt_base = r.get("gt_in_base", "").lower() in ("true", "1")
            gt_filtered = r.get("gt_in_filtered", "").lower() in ("true", "1")
            gt_pref = r.get("gt_in_prefilter", "").lower() in ("true", "1")
            loc_type = (r.get("loc_type") or "UNKNOWN").strip()
            etat = (r.get("sirene_etat") or "UNKNOWN").strip()
            reason = (r.get("loss_reason") or "").strip()

            if gt_base:
                self.gt_in_base_count += 1
            if gt_filtered:
                self.gt_in_filtered_count += 1
            if gt_pref:
                self.gt_in_prefilter_count += 1
            if reason:
                self.loss_reasons[reason] += 1

            if loc_type not in self.by_loc_type:
                self.by_loc_type[loc_type] = {"total": 0, "base": 0, "prefilter": 0}
            self.by_loc_type[loc_type]["total"] += 1
            if gt_base:
                self.by_loc_type[loc_type]["base"] += 1
            if gt_pref:
                self.by_loc_type[loc_type]["prefilter"] += 1

            if etat not in self.by_etat:
                self.by_etat[etat] = {"total": 0, "base": 0, "prefilter": 0}
            self.by_etat[etat]["total"] += 1
            if gt_base:
                self.by_etat[etat]["base"] += 1
            if gt_pref:
                self.by_etat[etat]["prefilter"] += 1

        # If deduplication occurred, rewrite clean file atomically
        total_file_lines = self._count_file_lines(path)
        if len(unique_rows) != total_file_lines - 1:
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(unique_rows)
            os.replace(tmp_path, path)
            logger.info("[CHECKPOINT] Deduplicated details CSV: %d unique queries preserved.", len(unique_rows))

        logger.info(
            "[CHECKPOINT] Synchronized state from details CSV: %d queries completed (Base: %.2f%%, Prefilter: %.2f%%)",
            self.total_queries,
            (self.gt_in_base_count / max(self.total_queries, 1)) * 100,
            (self.gt_in_prefilter_count / max(self.total_queries, 1)) * 100,
        )
        return seen_ids

    def init_details_writer(self, details_path: Path, fieldnames: List[str], resume: bool = False) -> None:
        """Initialize streaming details CSV writer with atomic header creation."""
        self._details_path = Path(details_path)
        self._details_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self._details_path.exists() and self._details_path.stat().st_size > 0
        mode = "a" if (resume and file_exists) else "w"

        self._details_file_handle = open(self._details_path, mode, encoding="utf-8", newline="")
        self._details_csv_writer = csv.DictWriter(
            self._details_file_handle,
            fieldnames=fieldnames,
            delimiter=";",
            extrasaction="ignore"
        )

        if not resume or not file_exists:
            self._details_csv_writer.writeheader()
            self._details_file_handle.flush()

    def record_query(
        self,
        idx: int,
        gt_in_base: bool,
        gt_in_filtered: bool,
        gt_in_prefilter: bool,
        loc_type: str,
        etat: str,
        loss_reason: str,
        detail_row: Optional[Dict[str, Any]] = None,
        elapsed_delta: float = 0.0,
    ) -> None:
        """Record the evaluation result of a single query."""
        self.last_processed_idx = idx
        self.total_queries += 1
        self.elapsed_seconds += max(0.0, elapsed_delta)

        if gt_in_base:
            self.gt_in_base_count += 1
        if gt_in_filtered:
            self.gt_in_filtered_count += 1
        if gt_in_prefilter:
            self.gt_in_prefilter_count += 1
        if loss_reason:
            self.loss_reasons[loss_reason] += 1

        # Stratified by loc_type
        if loc_type not in self.by_loc_type:
            self.by_loc_type[loc_type] = {"total": 0, "base": 0, "prefilter": 0}
        self.by_loc_type[loc_type]["total"] += 1
        if gt_in_base:
            self.by_loc_type[loc_type]["base"] += 1
        if gt_in_prefilter:
            self.by_loc_type[loc_type]["prefilter"] += 1

        # Stratified by etat
        if etat not in self.by_etat:
            self.by_etat[etat] = {"total": 0, "base": 0, "prefilter": 0}
        self.by_etat[etat]["total"] += 1
        if gt_in_base:
            self.by_etat[etat]["base"] += 1
        if gt_in_prefilter:
            self.by_etat[etat]["prefilter"] += 1

        # Stream detail row to CSV
        if detail_row and self._details_csv_writer and self._details_file_handle:
            self._details_csv_writer.writerow(detail_row)
            self._details_file_handle.flush()

        self._unsaved_updates += 1
        if self._unsaved_updates >= self.save_interval:
            self.save(status="IN_PROGRESS")

    def save(self, status: str = "IN_PROGRESS", force: bool = False) -> None:
        """Atomically persist current state to disk."""
        if not force and self._unsaved_updates == 0 and status == self.status:
            return

        self.status = status
        now_iso = datetime.now(timezone.utc).isoformat()

        payload = {
            "tag": self.tag,
            "status": self.status,
            "crm_path": self.crm_path_str,
            "config_signature": self.config_signature,
            "total_rows": self.total_rows,
            "last_processed_idx": self.last_processed_idx,
            "total_queries": self.total_queries,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "gt_in_base_count": self.gt_in_base_count,
            "gt_in_filtered_count": self.gt_in_filtered_count,
            "gt_in_prefilter_count": self.gt_in_prefilter_count,
            "by_loc_type": self.by_loc_type,
            "by_etat": self.by_etat,
            "loss_reasons": dict(self.loss_reasons),
            "last_saved_at": now_iso,
        }

        tmp_file = self.checkpoint_dir / f"checkpoint_{self.tag}.json.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            if self.checkpoint_file.exists():
                try:
                    # Keep one backup
                    if self.backup_file.exists():
                        self.backup_file.unlink()
                    self.checkpoint_file.replace(self.backup_file)
                except Exception:
                    pass

            tmp_file.replace(self.checkpoint_file)
            self._unsaved_updates = 0
        except Exception as e:
            logger.error("Failed to save checkpoint atomically: %s", e)

    def finalize_completed(self, summary: Dict[str, Any]) -> None:
        """Mark run as completed and archive checkpoint so future runs start clean."""
        self.save(status="COMPLETED", force=True)
        if self._details_file_handle:
            try:
                self._details_file_handle.flush()
                self._details_file_handle.close()
            except Exception:
                pass
            self._details_file_handle = None

        # Archive active checkpoint to completed marker and remove active checkpoint
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.replace(self.completed_marker)
            if self.backup_file.exists():
                self.backup_file.unlink()
            logger.info("Checkpoint finalized and archived to %s", self.completed_marker)
        except Exception as e:
            logger.warning("Could not archive completed checkpoint: %s", e)

    def close(self) -> None:
        """Flush files and close handles."""
        self.save(force=True)
        if self._details_file_handle:
            try:
                self._details_file_handle.flush()
                self._details_file_handle.close()
            except Exception:
                pass
            self._details_file_handle = None
