REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V2.1 — Architecture Restructuring
REM  STATUS  : Incremental V2.1 inference script. First structured dual-engine build.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.1 (Location & Prefixed Retrieval)
echo =======================================================================
echo     SIRETO MACHINE LEARNING MATCHER VERSION 2.1 (LOCATION FIRST)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data/crm_ok_gt.csv]: "
if "%crm_path%"=="" set crm_path=data/crm_ok_gt.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_2.1.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_2.1.xlsx

echo.
echo Launching Version 2.1 Location-First Inference Engine...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v21.py --crm-path "%crm_path%" --output-path "%output_path%"

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
