@echo off
color 0E
title Version 4.0 — 1,000 Sample Resume Test (Seed 51816)

echo ===============================================================================
echo       VERSION 4.0 RESUME SAMPLE TEST (1,000 SAMPLES — SEED 51816)
echo ===============================================================================
echo Output File: %USERPROFILE%\Desktop\results_1000_sample_v40_random.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scratch\run_sample_v40_test_fixed_51816.py"

echo.
echo ===============================================================================
echo TEST COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_1000_sample_v40_random.xlsx
echo ===============================================================================
pause
