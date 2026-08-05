REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V2.6 — Standard Sample
REM  STATUS  : Standard 1000-row sample validation run for V2.6.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher 1,000 Sample Test (Version 2.6)
echo =======================================================================
echo     SIRETO ML MATCHER 1,000 SAMPLE BENCHMARK TEST (VERSION 2.6)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

echo Launching 1,000-Sample Benchmark Test on Version 2.6...
echo.

.\venv\Scripts\python.exe C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v26_test.py

echo.
echo =======================================================================
echo Benchmark Execution Complete! Results saved to Desktop.
echo =======================================================================
pause
