@echo off
color 0D
title Version 4.2 – 1,000 Sample Random Test

echo ===============================================================================
echo       VERSION 4.2 PARALLEL DYNAMIC RANDOM SAMPLE TEST (1,000 SAMPLES)
echo ===============================================================================
echo Output File: %USERPROFILE%\Desktop\results_1000_sample_v42_random.xlsx
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set XGB_INFER_WORKERS=2

"C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\venv\Scripts\python.exe" "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\scratch\run_sample_v42_test_random.py"

echo.
echo ===============================================================================
echo TEST COMPLETE! Output saved to: %USERPROFILE%\Desktop\results_1000_sample_v42_random.xlsx
echo ===============================================================================
pause
