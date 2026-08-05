REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V1.0 — Results Modification
REM  STATUS  : First versioned optimised runner. Adds result post-processing.
REM ============================================================

@echo off
title Sireto - ML Full Pipeline Optimized 1.0 (xgb_matcher_opt)
echo ==========================================================
echo Starting Optimized XGBoost Two-Stage ML Matcher on Full Dataset (v1.0)...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_opt.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running Optimized ML pipeline (using local partitions + fallback)...
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_opt.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full_ml_opt.csv --allow-no-semantic

if %ERRORLEVEL% NEQ 0 (
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
echo [SUCCESS] Full Optimized ML execution completed!
echo Output saved to Desktop: results_full_ml_opt.xlsx
echo ==========================================================
pause
