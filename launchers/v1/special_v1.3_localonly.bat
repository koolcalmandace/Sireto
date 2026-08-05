REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V1.3 — Local-Only Mode
REM  STATUS  : Local-only mode pinned to V1.3 inference script.
REM ============================================================

@echo off
echo ===================================================
echo RUNNING SIRETO MODEL VERSION 1.3 (LOCAL-ONLY PUR OFFLINE)
echo Features:
echo   - Restricted Postcode-INSEE Fallback
echo   - City-to-INSEE Cascading
echo   - Short Acronym Safeguard (len ^<= 4)
echo   - Public Entity Department Gate
echo   - Pure Offline Engine (NO Web Scraper)
echo Speed: ~2-3 minutes for 17,000 lines
echo ===================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
call .\venv\Scripts\activate.bat
python scripts/infer_xgb_two_stage_v13.py --crm-path data/full_crm_test.parquet --output-path reports/results_full_ml_opt_local_1.3.csv --disable-scraper
pause
