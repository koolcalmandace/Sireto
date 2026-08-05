REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V2.0 — Architecture Restructuring
REM  STATUS  : Dual-engine inference introduced. Interactive CRM/output path prompts.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.0 (Full Opt 2.0)
echo =======================================================================
echo          SIRETO MACHINE LEARNING MATCHER VERSION 2.0 (FULL OPT 2.0)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data_v1.1/CRM_database.csv]: "
if "%crm_path%"=="" set crm_path=data_v1.1/CRM_database.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_2.0.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_2.0.xlsx

echo.
echo Launching Version 2.0 Dual-Engine Inference Engine...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v20.py --crm-path "%crm_path%" --output-path "%output_path%"

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
