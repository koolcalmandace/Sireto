REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.2 — Retrieval Restructuring
REM  STATUS  : V3.2 with improved candidate pool quality.
REM ============================================================

@echo off
TITLE Sireto Version 3.2 Production Matching Runner
COLOR 0A
echo =======================================================================
echo     SIRETO MATCHING ENGINE - VERSION 3.2 PRODUCTION RUNNER
echo =======================================================================
echo.
echo Launching Version 3.2 Production Matching Engine (17,054 records)...
echo Foundation: V3.1 Stable JSONL Progress + Street Abbreviation Jaro Normalization
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" scripts/infer_xgb_two_stage_v32.py --input-file "data/crm_ok_gt.csv" --output-file "C:\Users\Kabouassi\Desktop\results_full_17k_v32.xlsx"

echo.
echo =======================================================================
echo MATCHING COMPLETE! Saved output to Desktop: results_full_17k_v32.xlsx
echo =======================================================================
pause
