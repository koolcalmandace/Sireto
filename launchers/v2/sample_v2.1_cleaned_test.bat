REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V2.1 — Cleaned Sample
REM  STATUS  : 1000-row sample test against the cleaned V2.1 inference script.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher 1,000 Sample Test (Version 2.1 Cleaned)
echo =======================================================================
echo     SIRETO ML MATCHER 1,000 SAMPLE BENCHMARK TEST (VERSION 2.1 CLEANED)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

echo Launching 1,000-Sample Benchmark Test on Version 2.1 Cleaned...
echo.

.\venv\Scripts\python.exe C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v21_cleaned_test.py

echo.
echo =======================================================================
echo Benchmark Preparation Check Complete!
echo =======================================================================
pause
