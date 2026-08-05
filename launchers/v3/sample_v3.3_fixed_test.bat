REM ============================================================
REM  TYPE    : SAMPLE TEST RUNNER
REM  VERSION : V3.3 — Fixed Seed
REM  STATUS  : Reproducible 1000-row sample for V3.3.
REM ============================================================

@echo off
TITLE Sireto Version 3.3 1,000-Sample Fixed Seed 42 Test
COLOR 0B
echo =======================================================================
echo SIRETO MATCHING ENGINE - VERSION 3.3 FIXED SEED 42 BENCHMARK
echo =======================================================================
echo.
echo Launching 1,000-Sample Fixed Benchmark Test (Fixed Seed: 42)...
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v33_test_fixed.py"

echo.
echo =======================================================================
echo BENCHMARK COMPLETE! Saved output to Desktop: results_1000_sample_v33_fixed.xlsx
echo =======================================================================
pause
