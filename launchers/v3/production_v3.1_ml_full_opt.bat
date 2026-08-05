REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.1 — Retrieval Restructuring
REM  STATUS  : V3.1 incremental refinements over 3-stream retrieval.
REM ============================================================

@echo off
TITLE Sireto Version 3.1 Production Matching Runner
COLOR 0A
echo =======================================================================
echo          SIRETO MATCHING ENGINE - VERSION 3.1 PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.1 Two-Stage XGBoost Matching Engine (17,054 records)...
echo Features: Compound Location Token Retrieval + Calibrated 0.85 Auto Cutoff
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v31.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v31.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v31.xlsx
echo =======================================================================
pause
