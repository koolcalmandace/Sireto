REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V3.4 — Retrieval Restructuring
REM  STATUS  : V3.4 with column key fixes and French street abbreviations.
REM ============================================================

@echo off
color 0A
title Version 3.4 Full Dual-Engine Production Matcher (V3.0 SSOT + V2.6 JSONL + Sister Resolution)

echo ===============================================================================
echo            SIRETO ENTERPRISE MATCHING PIPELINE — VERSION 3.4 FULL
echo ===============================================================================
echo Architecture: V3.0 SSOT 3-Stream Candidate Retrieval + V2.6 Progress JSONL
echo Enhancements: French Street Abbr Normalization + Active Sister Branch Tie-Breaker
echo Input File  : data\crm_ok_gt.csv
echo Output File : %USERPROFILE%\Desktop\results_full_v34.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scripts\infer_xgb_two_stage_v34.py" --input-file "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv" --output-file "%USERPROFILE%\Desktop\results_full_v34.xlsx"

echo.
echo ===============================================================================
echo MATCHING COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_full_v34.xlsx
echo ===============================================================================
pause
