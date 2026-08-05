@echo off
title Sireto - ML Full Pipeline Version 4.2 (xgb_matcher_v42)
echo ==========================================================
echo Starting XGBoost Two-Stage ML Matcher Version 4.2...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\Desktop\results_full_ml_v42.xlsx
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo Running Version 4.2 ML pipeline (using parallel workers)...
set PYTHONUNBUFFERED=1
:: Run the parallel inference engine which will automatically load the latest V4.2 trained model
.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v41.py --input-file data/crm_ok_gt.csv --output-file data/reports_normalized/results_full_ml_v42.csv

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Version 4.2 ML Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full_ml_v42.csv').to_excel('C:/Users/Kabouassi/Desktop/results_full_ml_v42.xlsx', index=False); print('Created results_full_ml_v42.xlsx on Desktop')"

echo.
echo ==========================================================
echo [SUCCESS] Version 4.2 Full ML execution completed!
echo Output saved to Desktop: results_full_ml_v42.xlsx
echo ==========================================================
pause
