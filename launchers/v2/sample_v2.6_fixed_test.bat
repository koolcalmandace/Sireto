REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V2.6 — Fixed Seed Sample
REM  STATUS  : 1000-row fixed-seed sample test for V2.6 — reproducible results.
REM ============================================================

@echo off
TITLE Sireto Machine Learning Matcher - 1,000 FIXED Sample Benchmark (Seed 42)
echo =======================================================================
echo     SIRETO ML MATCHER - 1,000 FIXED SAMPLE BENCHMARK (VERSION 2.6)
echo =======================================================================
echo.
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

echo Launching 1,000-Sample FIXED Seed 42 A/B Benchmark Test on Version 2.6...
echo.

.\venv\Scripts\python.exe C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v26_test_fixed.py

echo.
echo =======================================================================
echo Fixed Seed Benchmark Complete! Output saved to: results_1000_sample_v26_fixed.xlsx
echo =======================================================================
pause
