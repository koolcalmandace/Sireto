REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V1.1 — Results Modification
REM  STATUS  : Incremental fix over V1.0 output handling.
REM ============================================================

@echo off
title Sireto - ML Full Pipeline Optimized 1.1 (xgb_matcher_opt)
echo ==========================================================
echo Starting Optimized XGBoost Two-Stage ML Matcher on Full Dataset (v1.1)...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Cache DB: data_v1.1\sirene_cache.sqlite
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_opt_1.1.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running Optimized ML pipeline with v1.1 Cache and Acronym/Public Gates...
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_opt.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full_ml_opt_1.1.csv --database-path data_v1.1/sirene_cache.sqlite --allow-no-semantic

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Optimized ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml_opt_1.1.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml_opt_1.1.xlsx', index=False); print('Created results_full_ml_opt_1.1.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Full Optimized ML 1.1 execution completed!
echo Output saved to Desktop: results_full_ml_opt_1.1.xlsx
echo ==========================================================
pause
