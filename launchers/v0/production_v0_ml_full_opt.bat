REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V0 — Baseline Optimised
REM  STATUS  : First optimised variant of the baseline ML runner.
REM ============================================================

@echo off
title Sireto - ML Full Pipeline Optimized (xgb_matcher)
echo ==========================================================
echo Starting OPTIMIZED XGBoost Two-Stage ML Matcher...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_opt.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running OPTIMIZED ML pipeline (using local partitions)...
set XGB_SEMANTIC_GATE_ENABLED=1
set PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_opt.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full_ml_opt.csv --allow-no-semantic --chunk-size 500

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo.
    echo [ERROR] Optimized ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml_opt.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml_opt.xlsx', index=False); print('Created results_full_ml_opt.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Optimized Full ML execution completed!
echo Output saved to Desktop: results_full_ml_opt.xlsx
echo ==========================================================
pause
