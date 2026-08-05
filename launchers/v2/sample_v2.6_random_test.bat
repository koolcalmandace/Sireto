REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V2.6 — Random Seed Sample
REM  STATUS  : 1000-row random-seed sample test for V2.6 — coverage testing.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher - 1,000 DYNAMIC RANDOM Sample Test
echo =======================================================================
echo     SIRETO ML MATCHER - 1,000 DYNAMIC RANDOM SAMPLE TEST (VERSION 2.6)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

echo Launching 1,000-Sample DYNAMIC FRESH RANDOM Test on Version 2.6...
echo.

.\venv\Scripts\python.exe C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v26_test_random.py

echo.
echo =======================================================================
echo Dynamic Random Test Complete! Output saved to: results_1000_sample_v26_random.xlsx
echo =======================================================================
pause
