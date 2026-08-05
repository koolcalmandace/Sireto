REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V1 — Local-Only Mode
REM  STATUS  : Runs inference using local partitions only — no live API calls.
REM ============================================================

@echo off
title Sireto - ML Full Pipeline (Local Only - No Scraper)
echo ==========================================================
echo Starting Pure Local XGBoost ML Matcher (Offline Engine)...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Cache DB: data_v1.1\sirene_cache.sqlite
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_opt_local_only.xlsx
echo Scraper: DISABLED (100%% Local ML & SQLite Matching)
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running Local-Only ML pipeline...
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_opt.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full_ml_opt_local_only.csv --database-path data_v1.1/sirene_cache.sqlite --allow-no-semantic --disable-scraper

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Local-Only ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml_opt_local_only.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml_opt_local_only.xlsx', index=False); print('Created results_full_ml_opt_local_only.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Local-Only ML execution completed!
echo Output saved to Desktop: results_full_ml_opt_local_only.xlsx
echo ==========================================================
pause
