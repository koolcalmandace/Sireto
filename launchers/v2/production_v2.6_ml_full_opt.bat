REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V2.6 — Architecture Restructuring
REM  STATUS  : Final V2 series production runner. Most stable V2 build.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.6 (Full Opt 2.6)
echo =======================================================================
echo     SIRETO ML MATCHER VERSION 2.6 (FULL OPT 2.6 - PURE OFFLINE)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data/crm_ok_gt.csv]: "
if "%crm_path%"=="" set crm_path=data/crm_ok_gt.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_2.6.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_2.6.xlsx

echo.
echo Launching Version 2.6 Full Production Inference Engine...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v26.py --crm-path "%crm_path%" --output-path "%output_path%"

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
