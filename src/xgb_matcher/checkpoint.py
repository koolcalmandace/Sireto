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
