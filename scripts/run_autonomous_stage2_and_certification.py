"""
scripts/run_autonomous_stage2_and_certification.py
Orchestrates the autonomous execution of:
  1. Stage 2 Unified Decider completion (unattended mode, ~20 min remaining).
  2. Movement & Quality Gate Analysis (Hit@1, Brier, Margins).
  3. Risk Model evaluation setup for 32k.
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_step(cmd_list, description):
    print("=" * 80)
    print(f"[AUTONOMOUS ORCHESTRATOR] Starting: {description}")
    print(f"Command: {' '.join(cmd_list)}")
    print("=" * 80, flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {res.returncode} after {elapsed:.1f}s", flush=True)
        sys.exit(res.returncode)
    print(f"\n[SUCCESS] Completed in {elapsed:.1f}s\n", flush=True)

if __name__ == "__main__":
    # Step 1: Finish Stage 2 Training in unattended mode
    cmd_s2 = [
        sys.executable,
        "scripts/train_unified_xgb_decider.py",
        "--sample-size", "10000",
        "--random-seed", "42",
        "--chunk-size", "250",
        "--resume",
        "--unattended",
    ]
    run_step(cmd_s2, "Stage 2 Retraining & Test Set Certification (Unattended)")

    print("\n" + "=" * 80)
    print("[AUTONOMOUS ORCHESTRATOR] STAGE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80, flush=True)
