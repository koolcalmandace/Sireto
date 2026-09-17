"""
test_checkpoint_and_power.py — Unit tests for CheckpointManager and PowerMonitor.
"""

import csv
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from src.xgb_matcher.checkpoint import CheckpointManager
from src.xgb_matcher.power_guard import PowerMonitor


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_lifecycle(temp_dir):
    """Test full checkpoint lifecycle: start, save, crash simulation, resume, and finalize."""
    tag = "test_run"
    crm_fake = temp_dir / "crm.csv"
    crm_fake.write_text("id;nom\n1;test", encoding="utf-8")

    ckpt_dir = temp_dir / "checkpoints"
    details_csv = temp_dir / "details.csv"
    fieldnames = ["crm_id", "gt_in_base", "loss_reason"]

    # Phase 1: Process 5 queries then simulate crash
    mgr1 = CheckpointManager(
        tag=tag,
        checkpoint_dir=ckpt_dir,
        total_rows=10,
        crm_path=crm_fake,
        save_interval=2,
    )
    assert not mgr1.is_resumable()

    mgr1.init_details_writer(details_csv, fieldnames=fieldnames, resume=False)

    for i in range(5):
        mgr1.record_query(
            idx=i,
            gt_in_base=(i % 2 == 0),
            gt_in_filtered=(i % 2 == 0),
            gt_in_prefilter=True,
            loc_type="INSEE",
            etat="ACTIF",
            loss_reason="" if (i % 2 == 0) else "NOT_IN_BASE",
            detail_row={"crm_id": f"ID_{i}", "gt_in_base": i % 2 == 0, "loss_reason": ""},
            elapsed_delta=0.1,
        )

    # Force save before simulated crash
    mgr1.save(status="PAUSED", force=True)
    mgr1.close()

    # Verify checkpoint file exists and is resumable
    assert mgr1.is_resumable()

    # Phase 2: Resume in new process/instance
    mgr2 = CheckpointManager(
        tag=tag,
        checkpoint_dir=ckpt_dir,
        total_rows=10,
        crm_path=crm_fake,
        save_interval=2,
    )
    assert mgr2.is_resumable()
    loaded = mgr2.load()
    assert loaded is True
    assert mgr2.last_processed_idx == 4
    assert mgr2.total_queries == 5
    assert mgr2.gt_in_base_count == 3  # 0, 2, 4
    assert mgr2.gt_in_prefilter_count == 5

    # Reopen CSV in append mode
    mgr2.init_details_writer(details_csv, fieldnames=fieldnames, resume=True)

    # Process remaining queries 5 to 9
    for i in range(5, 10):
        mgr2.record_query(
            idx=i,
            gt_in_base=True,
            gt_in_filtered=True,
            gt_in_prefilter=True,
            loc_type="INSEE",
            etat="ACTIF",
            loss_reason="",
            detail_row={"crm_id": f"ID_{i}", "gt_in_base": True, "loss_reason": ""},
            elapsed_delta=0.1,
        )

    summary = {"total_queries": mgr2.total_queries, "recall": 80.0}
    mgr2.finalize_completed(summary)

    # Check details CSV row count (1 header + 10 rows = 11 lines)
    with open(details_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f, delimiter=";"))
    assert len(reader) == 11
    assert reader[0] == fieldnames
    assert reader[1][0] == "ID_0"
    assert reader[10][0] == "ID_9"

    # After completion, checkpoint should not be resumable
    mgr3 = CheckpointManager(
        tag=tag,
        checkpoint_dir=ckpt_dir,
        total_rows=10,
        crm_path=crm_fake,
    )
    assert not mgr3.is_resumable()


def test_power_monitor_pause_and_resume():
    """Test PowerMonitor pause triggering, state queries, and wait/resume."""
    pause_called = []
    resume_called = []

    pm = PowerMonitor(
        on_pause_callback=lambda: pause_called.append(True),
        on_resume_callback=lambda: resume_called.append(True),
    )
    assert pm.pause_on_lock is True

    with pm:
        assert not pm.should_pause()

        # Trigger pause
        pm.trigger_pause("Test Pause")
        assert pm.should_pause()
        assert len(pause_called) == 1

        # Test wait_for_resume in background thread
        resumed_status = []

        def worker():
            ok = pm.wait_for_resume(timeout=5.0)
            resumed_status.append(ok)

        t = threading.Thread(target=worker)
        t.start()

        time.sleep(0.2)
        assert t.is_alive()  # Still waiting

        # Resume
        pm.trigger_resume("Test Resume")
        t.join(timeout=2.0)
        assert not t.is_alive()
        assert resumed_status == [True]
        assert not pm.should_pause()
        assert len(resume_called) == 1


def test_power_monitor_inactivity_timeout():
    """Test that idle timeout triggers pause automatically when user is inactive."""
    pm = PowerMonitor(
        pause_on_lock=False,
        idle_timeout_seconds=0.1,  # Fast trigger for test
        poll_interval=0.05,
    )
    # Mock idle seconds return
    pm._get_windows_idle_seconds = lambda: 0.2

    with pm:
        time.sleep(0.15)
        assert pm.should_pause()
        assert "User Inactive" in pm._pause_reason

        # Simulate user activity resumption
        pm._get_windows_idle_seconds = lambda: 0.0
        time.sleep(0.15)
        assert not pm.should_pause()


def test_power_monitor_exit_trigger():
    """Test that trigger_exit sets should_exit and executes callback."""
    exit_called = []
    pm = PowerMonitor(
        on_exit_callback=lambda: exit_called.append(True),
    )
    assert not pm.should_exit()
    assert not pm.is_exit_requested

    pm.trigger_exit("System Shutdown Test")
    assert pm.should_exit()
    assert pm.is_exit_requested
    assert not pm.should_pause()
    assert len(exit_called) == 1
    # When exit is requested, wait_for_resume returns False immediately
    assert pm.wait_for_resume(timeout=1.0) is False


def test_power_monitor_wait_for_resume_aborts_on_exit():
    """Test that a paused thread waiting for resume unblocks immediately when exit/shutdown occurs."""
    pm = PowerMonitor()
    with pm:
        pm.trigger_pause("Temporary Pause")
        assert pm.should_pause()

        wait_result = []

        def worker():
            ok = pm.wait_for_resume(timeout=5.0)
            wait_result.append(ok)

        t = threading.Thread(target=worker)
        t.start()

        time.sleep(0.2)
        assert t.is_alive()  # Still waiting for resume

        # Now simulate an OS shutdown triggering terminal exit
        t_start = time.time()
        pm.trigger_exit("OS Shutdown")
        t.join(timeout=2.0)
        elapsed = time.time() - t_start

        # Thread must have unblocked immediately (< 1s), well before the 5.0s timeout!
        assert not t.is_alive()
        assert elapsed < 1.5
        assert wait_result == [False]
        assert pm.should_exit()


def test_checkpoint_sync_from_details_csv(temp_dir):
    """Test sync_from_details_csv correctly deduplicates and reconstructs checkpoint stats."""
    details_csv = temp_dir / "details.csv"
    fieldnames = ["crm_id", "loc_type", "sirene_etat", "gt_in_base", "gt_in_filtered", "gt_in_prefilter", "loss_reason"]

    # Write synthetic CSV with a duplicate crm_id (ID_1 appears twice)
    with open(details_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerow({"crm_id": "ID_1", "loc_type": "INSEE", "sirene_etat": "A", "gt_in_base": "True", "gt_in_filtered": "True", "gt_in_prefilter": "True", "loss_reason": ""})
        writer.writerow({"crm_id": "ID_2", "loc_type": "INSEE", "sirene_etat": "A", "gt_in_base": "True", "gt_in_filtered": "True", "gt_in_prefilter": "False", "loss_reason": "PRUNED"})
        writer.writerow({"crm_id": "ID_1", "loc_type": "INSEE", "sirene_etat": "A", "gt_in_base": "True", "gt_in_filtered": "True", "gt_in_prefilter": "True", "loss_reason": ""})

    mgr = CheckpointManager(tag="test_sync", checkpoint_dir=temp_dir / "ckpts", total_rows=10)
    seen_ids = mgr.sync_from_details_csv(details_csv, fieldnames)

    assert seen_ids == {"ID_1", "ID_2"}
    assert mgr.total_queries == 2
    assert mgr.gt_in_base_count == 2
    assert mgr.gt_in_prefilter_count == 1
    assert mgr.loss_reasons["PRUNED"] == 1


def test_cluster_rows_by_locality():
    """Test in-Python geographic clustering groups rows with same insee/postcode together."""
    from scripts.evaluate_geo_resolution import cluster_rows_by_locality

    raw_rows = [
        {"crm_id": "1", "crm_insee": "75101", "crm_cp": "75001"},
        {"crm_id": "2", "crm_insee": "13001", "crm_cp": "13001"},
        {"crm_id": "3", "crm_insee": "75101", "crm_cp": "75001"},
        {"crm_id": "4", "crm_insee": "69001", "crm_cp": "69001"},
        {"crm_id": "5", "crm_insee": "75101", "crm_cp": "75001"},
    ]

    clustered = cluster_rows_by_locality(raw_rows)
    clustered_insee = [r["crm_insee"] for r in clustered]

    # Both Marseille and Lyon are grouped, and all three Paris 75101 queries are adjacent
    assert clustered_insee == ["13001", "69001", "75101", "75101", "75101"]



