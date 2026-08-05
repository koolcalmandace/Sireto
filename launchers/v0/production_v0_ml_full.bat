REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V0 — Baseline
REM  STATUS  : Original ML two-stage inference on full dataset. No optimisations.
REM ============================================================

@echo off
title Sireto - ML Full Pipeline (xgb_matcher)
echo ==========================================================
echo Starting XGBoost Two-Stage ML Matcher on Full Dataset...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running ML pipeline (using local partitions)...
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full_ml.csv --allow-no-semantic

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml.xlsx', index=False); print('Created results_full_ml.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Full ML execution completed!
echo Output saved to Desktop: results_full_ml.xlsx
echo ==========================================================
pause
