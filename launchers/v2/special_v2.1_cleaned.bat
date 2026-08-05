REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V2.1 — Cleaned Full Run
REM  STATUS  : Full production run using the cleaned/refactored V2.1 inference script.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher Version 2.1 Cleaned (Hierarchical 16-Vector)
echo =======================================================================
echo     SIRETO MACHINE LEARNING MATCHER VERSION 2.1 CLEANED (16 VECTORS)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

set /p crm_path="Enter path to CRM Database [Default: data/crm_ok_gt.csv]: "
if "%crm_path%"=="" set crm_path=data/crm_ok_gt.csv

set /p output_path="Enter path for output results [Default: C:\Users\Kabouassi\Desktop\results_full_opt_2.1_cleaned.xlsx]: "
if "%output_path%"=="" set output_path=C:\Users\Kabouassi\Desktop\results_full_opt_2.1_cleaned.xlsx

echo.
echo Launching Version 2.1 Cleaned Inference Engine...
echo Input:  %crm_path%
echo Output: %output_path%
echo.

.\venv\Scripts\python.exe scripts/infer_xgb_two_stage_v21_cleaned.py --crm-path "%crm_path%" --output-path "%output_path%"

echo.
echo =======================================================================
echo Inference Complete! Results saved to: %output_path%
echo =======================================================================
pause
