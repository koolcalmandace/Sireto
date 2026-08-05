REM ============================================================
REM  TYPE    : PRODUCTION RUNNER
REM  VERSION : V1.3 — Results Modification
REM  STATUS  : Final V1 series production runner. Consolidates all V1 fixes.
REM ============================================================

@echo off
echo ===================================================
echo RUNNING SIRETO MODEL VERSION 1.3 (FULL DUAL-ENGINE)
echo Features:
echo   - Restricted Postcode-INSEE Fallback
echo   - City-to-INSEE Cascading
echo   - Short Acronym Safeguard (len ^<= 4)
echo   - Public Entity Department Gate
echo   - Web Scraper Fallback Engine
echo ===================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
call .\venv\Scripts\activate.bat
python scripts/infer_xgb_two_stage_v13.py --crm-path data/full_crm_test.parquet --output-path reports/results_full_ml_opt_1.3.csv
pause
