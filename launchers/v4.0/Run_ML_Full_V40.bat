@echo off
title Sireto - ML Full Pipeline Version 4.0 (xgb_matcher_v40)
echo ==========================================================
echo Starting XGBoost Two-Stage ML Matcher Version 4.0...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_v40.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo Running Version 4.0 ML pipeline (using local partitions)...
set PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v40.py --input-file data/crm_ok_gt.csv --output-file data/reports_normalized/results_full_ml_v40.csv

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Version 4.0 ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml_v40.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml_v40.xlsx', index=False); print('Created results_full_ml_v40.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Version 4.0 Full ML execution completed!
echo Output saved to Desktop: results_full_ml_v40.xlsx
echo ==========================================================
pause
