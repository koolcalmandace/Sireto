REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V3.0 — Fixed Seed
REM  STATUS  : Reproducible 1000-row sample for V3.0.
REM ============================================================

@echo off
TITLE Sireto Version 3.0 1,000-Sample Fixed Seed 42 Test
COLOR 0B
echo =======================================================================
echo     SIRETO MATCHING ENGINE - VERSION 3.0 FIXED SEED 42 BENCHMARK TEST
echo =======================================================================
echo.
echo Launching 1,000-Sample Fixed Benchmark Test (Fixed Seed: 42)...
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v30_test_fixed.py"

echo.
echo =======================================================================
echo BENCHMARK COMPLETE! Saved output to Desktop: results_1000_sample_v30_fixed.xlsx
echo =======================================================================
pause
