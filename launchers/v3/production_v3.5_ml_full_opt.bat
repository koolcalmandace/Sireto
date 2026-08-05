REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.5 — Retrieval Restructuring [LATEST]
REM  STATUS  : Latest production runner. V3.0 SSOT + robust keys + sister tie-breaker.
REM ============================================================

@echo off
color 0A
title Version 3.5 Full Dual-Engine Production Matcher (V3.0 SSOT + Robust Keys + Sister Resolution)

echo ===============================================================================
echo            SIRETO ENTERPRISE MATCHING PIPELINE — VERSION 3.5 FULL
echo ===============================================================================
echo Architecture: V3.0 SSOT 3-Stream Candidate Retrieval + V2.6 Progress JSONL
echo Enhancements: Complete Column Keys + French Street Abbr + Active Sister Tie-Breaker
echo Input File  : data\crm_ok_gt.csv
echo Output File : %USERPROFILE%\Desktop\results_full_v35.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\infer_xgb_two_stage_v35.py" --input-file "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv" --output-file "%USERPROFILE%\Desktop\results_full_v35.xlsx"

echo.
echo ===============================================================================
echo MATCHING COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_full_v35.xlsx
echo ===============================================================================
pause
