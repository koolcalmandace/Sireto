REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V3.5 — Random Seed
REM  STATUS  : Random 1000-row coverage test for V3.5 (latest).
REM ============================================================

@echo off
color 0E
title Version 3.5 — 1,000 Sample Dynamic Random Test

echo ===============================================================================
echo       VERSION 3.5 DYNAMIC RANDOM SAMPLE TEST (1,000 SAMPLES)
echo ===============================================================================
echo Output File: %USERPROFILE%\Desktop\results_1000_sample_v35_random.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scratch\run_sample_v35_test_random.py"

echo.
echo ===============================================================================
echo TEST COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_1000_sample_v35_random.xlsx
echo ===============================================================================
pause
