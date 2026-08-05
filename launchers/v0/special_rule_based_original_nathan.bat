REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V0 — Rule-Based (Nathan original)
REM  STATUS  : Original rule-based fallback pipeline — Nathan's reference copy.
REM ============================================================

@echo off
title Sireto - Rule-Based Full Pipeline (pipe_v6)
echo ==========================================================
echo Starting Rule-Based Fallback Pipeline on Full Dataset...
echo Source: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\crm_ok_gt.csv
echo Destination: C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto\data\reports_normalized\results_full.csv
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

echo.
echo Running pipeline (this may take a long time due to API limits)...
.\venv\Scripts\python.exe scripts/run_pipe_v6.py --crm-path data/crm_ok_gt.csv --output-path data/reports_normalized/results_full.csv --stats-json data/reports_normalized/stats_full.json

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline execution failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Converting results to Excel (.xlsx)...
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/reports_normalized/results_full.csv', sep=';').to_excel('data/reports_normalized/results_full.xlsx', index=False); print('Created results_full.xlsx')"

echo.
echo Copying results_full.xlsx to Desktop...
copy "data\reports_normalized\results_full.xlsx" "C:\Users\Kabouassi\Desktop\results_full_rule_based.xlsx"

echo.
echo ==========================================================
echo [SUCCESS] Full rule-based execution completed!
echo Output saved to Desktop: results_full_rule_based.xlsx
echo ==========================================================
pause
