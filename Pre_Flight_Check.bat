@echo off
color 0A
title Sireto — Pre-Flight Diagnostic Check

echo ===============================================================================
echo            SIRETO MATCHING ENGINE — PRE-FLIGHT DIAGNOSTIC SUITE
echo ===============================================================================
echo Running 5 rapid validation unit tests (< 2 minutes)...
echo ===============================================================================
echo.

cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set XGB_SEMANTIC_ENABLED=1
set XGB_INFER_WORKERS=2

".\venv\Scripts\python.exe" -u "scripts\pre_flight_check.py"

echo.
pause
