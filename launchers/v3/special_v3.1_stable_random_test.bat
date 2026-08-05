REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V3.1 — Stable (Random Seed)
REM  STATUS  : Stable-pinned V3.1 — random-seed 1000-row test.
REM ============================================================

@echo off
TITLE Sireto Version 3.1 Stable 1,000-Sample Dynamic Random Test
COLOR 0E
echo =======================================================================
echo SIRETO MATCHING ENGINE - VERSION 3.1 STABLE DYNAMIC RANDOM TEST
echo =======================================================================
echo.
echo Launching 1,000-Sample Fresh Dynamic Random Test...
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\brain\9618f677-ee27-4e6d-91e7-1640c72fca92\scratch\run_sample_v31_stable_test_random.py"

echo.
echo =======================================================================
echo RANDOM TEST COMPLETE! Saved output to Desktop: results_1000_sample_v31_stable_random.xlsx
echo =======================================================================
pause
