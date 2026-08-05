"""Pre-Flight Verification Script for Sireto Matching Pipeline.

Executes 5 rapid diagnostic unit checks (< 2 minutes total) to validate
model health, feature integrity, and version consistency before running
large production batches.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Project path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import numpy as np

from src.xgb_matcher.profile import InferenceProfile, ensure_semantic_enabled
from scripts.infer_xgb_two_stage_v41 import Version41XgbInferenceEngine


def run_preflight_checks() -> bool:
    print("=" * 70)
    print("       SIRETO MATCHING ENGINE -- PRE-FLIGHT DIAGNOSTIC SUITE")
    print("=" * 70)
    print()

    ensure_semantic_enabled()
    all_passed = True

    # -----------------------------------------------------------------------
    # TEST 1: Model Version & Active Pointer Consistency
    # -----------------------------------------------------------------------
    print("[TEST 1/5] Checking Active Model Pointer & Profile Resolution...")
    try:
        active_txt = Path("models/ACTIVE_VERSION.txt")
        expected_version = active_txt.read_text().strip() if active_txt.exists() else None
        profile = InferenceProfile.from_latest_meta(strict=False)
        print(f"  Active Pointer: {expected_version}")
        print(f"  Ranker Path:    {profile.ranker_fast_path}")
        print(f"  Decider Path:   {profile.decider_path}")
        print(f"  Stage 1 Top N:  {profile.stage1_top_n}")
        if expected_version and expected_version not in str(profile.decider_path):
            print(f"  [FAIL] Active pointer ({expected_version}) does not match loaded decider ({profile.decider_path})")
            all_passed = False
        else:
            print("  [PASS] Active model pointer correctly loaded.")
    except Exception as e:
        print(f"  [FAIL] Model profile resolution error: {e}")
        all_passed = False

    print()

    # -----------------------------------------------------------------------
    # TEST 2: Pipeline Initialization & Smoke Test
    # -----------------------------------------------------------------------
    print("[TEST 2/5] Initializing Two-Stage Engine (Smoke Test)...")
    try:
        engine = Version41XgbInferenceEngine.from_profile(profile)
        print("  [PASS] Inference Engine initialized successfully.")
    except Exception as e:
        print(f"  [FAIL] Engine initialization failed: {e}")
        return False

    print()

    # -----------------------------------------------------------------------
    # TEST 3: Known-Good Match Verification
    # -----------------------------------------------------------------------
    print("[TEST 3/5] Testing Known-Good Match Query...")
    sample_query_good = {
        "crm_name": "COVAGE",
        "crm_cp": "92100",
        "crm_commune": "BOULOGNE BILLANCOURT",
        "crm_adresse": "12 RUE DE LA REDOUTE",
    }
    try:
        res_good = engine.process_row(sample_query_good, top_n=profile.stage1_top_n)
        siret = res_good.get("v41_siret") or res_good.get("siret")
        score = res_good.get("v41_score") or res_good.get("score", 0.0)
        decision = res_good.get("v41_decision") or res_good.get("decision")
        print(f"  Match Result: SIRET={siret} | Score={score:.4f} | Decision={decision}")
        if decision == "AUTO" and siret:
            print("  [PASS] Known-good query matched automatically with high confidence.")
        else:
            print(f"  [WARN] Known-good query produced decision '{decision}' (Expected AUTO).")
    except Exception as e:
        print(f"  [FAIL] Known-good match error: {e}")
        all_passed = False

    print()

    # -----------------------------------------------------------------------
    # TEST 4: Known-No-Match Safety Verification
    # -----------------------------------------------------------------------
    print("[TEST 4/5] Testing Known-No-Match Query (Fictional Company)...")
    sample_query_bad = {
        "crm_name": "FICTIONAL IMPOSSIBLE COMPANY XYZ 9999",
        "crm_cp": "75001",
        "crm_commune": "PARIS",
        "crm_adresse": "999 RUE IMAGINAIRE",
    }
    try:
        res_bad = engine.process_row(sample_query_bad, top_n=profile.stage1_top_n)
        score_bad = res_bad.get("v41_score") or res_bad.get("score", 0.0)
        decision_bad = res_bad.get("v41_decision") or res_bad.get("decision")
        print(f"  No-Match Result: Score={score_bad:.4f} | Decision={decision_bad}")
        if decision_bad in ("NO_MATCH", "REVIEW") and score_bad < 0.85:
            print("  [PASS] Fictional company correctly rejected/routed (no false AUTO).")
        else:
            print(f"  [FAIL] Fictional company produced false AUTO match!")
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] Known-no-match query error: {e}")
        all_passed = False

    print()

    # -----------------------------------------------------------------------
    # TEST 5: Feature Completeness & NaN Sanity Check
    # -----------------------------------------------------------------------
    print("[TEST 5/5] Checking Feature Vector Completeness & NaN Sanity...")
    try:
        top_candidates = res_good.get("top_candidates", [])
        if top_candidates:
            first_cand = top_candidates[0]
            nan_feats = [k for k, v in first_cand.items() if isinstance(v, float) and np.isnan(v)]
            if nan_feats:
                print(f"  [FAIL] Found NaN values in candidate features: {nan_feats}")
                all_passed = False
            else:
                print(f"  [PASS] All {len(first_cand)} candidate features clean and non-NaN.")
        else:
            print("  [WARN] No candidate features returned to inspect.")
    except Exception as e:
        print(f"  [FAIL] Feature completeness check error: {e}")
        all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("     PRE-FLIGHT CHECKS COMPLETE: ALL SYSTEMS READY FOR PRODUCTION!")
    else:
        print("     PRE-FLIGHT CHECKS FAILED: PLEASE REVIEW ERRORS ABOVE BEFORE RUNNING.")
    print("=" * 70)
    print()

    return all_passed


if __name__ == "__main__":
    success = run_preflight_checks()
    sys.exit(0 if success else 1)
