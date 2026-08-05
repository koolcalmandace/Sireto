REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V2.2 — Architecture Restructuring
REM  STATUS  : V2.2 production runner with updated inference engine.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.2 (Full Opt 2.2)
echo =======================================================================
echo     SIRETO MACHINE LEARNING MATCHER VERSION 2.2 (FULL OPT 2.2)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data/crm_ok_gt.csv]: "
if "%crm_path%"=="" set crm_path=data/crm_ok_gt.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_2.2.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_2.2.xlsx

echo.
echo Launching Version 2.2 Full Production Inference Engine...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v22.py --crm-path "%crm_path%" --output-path "%output_path%"

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
