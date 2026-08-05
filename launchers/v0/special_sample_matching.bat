REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V0 — Sample Matching
REM  STATUS  : Baseline sample matching runner for quick validation.
REM ============================================================

@echo off
title Sireto - ML Sample Matching (500 Records)
echo ==========================================================
echo Starting OPTIMIZED Sample ML Matcher...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_sample_no_match.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_sample_no_match_opt.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running optimized sample ML pipeline...
set XGB_SEMANTIC_GATE_ENABLED=1
set PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_opt.py --crm-path data/crm_sample_no_match.csv --output-path data/reports_normalized/results_sample_no_match_opt.csv --allow-no-semantic --chunk-size 500

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Sample ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Exporting results to Excel on Desktop...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_sample_no_match_opt.csv').to_excel('C:/Users/Kabouassi/Desktop/results_sample_no_match_opt.xlsx', index=False); print('Created results_sample_no_match_opt.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Sample execution completed!
echo ==========================================================
pause
