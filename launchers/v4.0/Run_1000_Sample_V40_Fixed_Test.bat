@echo off
color 0B
title Version 4.0 — 1,000 Sample Fixed Benchmark Test (Seed 42)

echo ===============================================================================
echo       VERSION 4.0 FIXED BENCHMARK TEST (1,000 SAMPLES — SEED 42)
echo ===============================================================================
echo Output File: %USERPROFILE%\Desktop\results_1000_sample_v40_fixed.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scratch\run_sample_v40_test_fixed.py"

echo.
echo ===============================================================================
echo TEST COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_1000_sample_v40_fixed.xlsx
echo ===============================================================================
pause
