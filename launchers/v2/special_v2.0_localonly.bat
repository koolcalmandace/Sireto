REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V2.0 — Local-Only Mode
REM  STATUS  : V2.0 run restricted to local partition lookups only.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.0 (Pure Offline Local-Only)
echo =======================================================================
echo     SIRETO MACHINE LEARNING MATCHER VERSION 2.0 (PURE OFFLINE LOCAL-ONLY)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data_v1.1/CRM_database.csv]: "
if "%crm_path%"=="" set crm_path=data_v1.1/CRM_database.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_local_v20.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_local_v20.xlsx

echo.
echo Launching Version 2.0 Pure Offline XGBoost Inference Engine (~2-3 min)...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v20.py --crm-path "%crm_path%" --output-path "%output_path%" --disable-scraper

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
