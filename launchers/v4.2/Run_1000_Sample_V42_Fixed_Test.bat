@echo off
color 0B
title Version 4.2 – 1,000 Sample Fixed Test (Seed 42)

echo ===============================================================================
echo       VERSION 4.2 PARALLEL FIXED SAMPLE TEST (1,000 SAMPLES – SEED 42)
echo ===============================================================================
echo Output File: %USERPROFILE%\Desktop\results_1000_sample_v42_fixed.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set XGB_INFER_WORKERS=2

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scratch\run_sample_v42_test_fixed.py"

echo.
echo ===============================================================================
echo TEST COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_1000_sample_v42_fixed.xlsx
echo ===============================================================================
pause
