REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V3.1 — Restored Full Run
REM  STATUS  : Full production run using the restored V3.1 inference state.
REM ============================================================

@echo off
TITLE Sireto Version 3.1 Restored Production Matching Runner
COLOR 0A
echo =======================================================================
echo     SIRETO MATCHING ENGINE - VERSION 3.1 RESTORED PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.1 Restored Matching Engine (17,054 records)...
echo Features: V2.6 Sequential Processing + Top-20 Lock + Compound Tokens + 0.85 Cutoff
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v31_restored.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v31_restored.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v31_restored.xlsx
echo =======================================================================
pause
