REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.0 — Retrieval Restructuring
REM  STATUS  : 3-stream candidate retrieval + native UL_SIGLE + EPCI resolver.
REM ============================================================

@echo off
TITLE Sireto Version 3.0 Production Matching Runner
COLOR 0A
echo =======================================================================
echo          SIRETO MATCHING ENGINE - VERSION 3.0 PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.0 Two-Stage XGBoost Matching Engine (17,054 records)...
echo Features: 3-Stream Candidate Retrieval + Native UL_SIGLE + EPCI Resolver
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v30.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v30.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v30.xlsx
echo =======================================================================
pause
