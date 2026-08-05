REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.3 — Retrieval Restructuring
REM  STATUS  : V3.3 with extended feature set.
REM ============================================================

@echo off
TITLE Sireto Version 3.3 Production Matching Runner
COLOR 0A
echo =======================================================================
echo     SIRETO MATCHING ENGINE - VERSION 3.3 PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.3 Production Matching Engine (17,054 records)...
echo Features: EPCI Territory Candidate Fallback + PM_DIRIGEANT Manager Stream + V3.2 Street Normalization
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v33.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v33.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v33.xlsx
echo =======================================================================
pause
