REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V3.1 — Stable Full Run
REM  STATUS  : Full production run using the stable-pinned V3.1 inference state.
REM ============================================================

@echo off
TITLE Sireto Version 3.1 Stable Production Matching Runner
COLOR 0A
echo =======================================================================
echo     SIRETO MATCHING ENGINE - VERSION 3.1 STABLE PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.1 Stable Production Engine (17,054 records)...
echo Architecture: V2.0-V2.6 JSONL Append & Resume + V3.1 Rules (Top-20 Lock)
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v31_stable.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v31_stable.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v31_stable.xlsx
echo =======================================================================
pause
