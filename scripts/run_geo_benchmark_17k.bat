@echo off
REM ============================================================
REM run_geo_benchmark_17k.bat
REM Evaluation de la resolution Geo & Recall sur le baseline 17 054
REM Branche : feature/geo-resolution-and-crm-expansion
REM Date    : 2026-08-21
REM ============================================================

SETLOCAL

SET PYTHON=py
SET LOG_DIR=logs\geo_validation
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK GEO-RESOLUTION (17 054 CRM Baseline)
echo ============================================================
%PYTHON% scripts\evaluate_geo_resolution.py ^
    --crm data\crm_ok_gt.csv ^
    --partitions-dir data\candidates_v7_all ^
    --output-dir reports\geo_validation ^
    --tag 17k_baseline ^
    --prefilter-k 500 ^
    --log-file %LOG_DIR%\run_geo_benchmark_17k.log

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] L'evaluation a rencontre une erreur.
    pause
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Evaluation terminee avec succes !
echo Rapports generes :
echo   - Resume JSON : reports\geo_validation\geo_benchmark_17k_baseline_summary.json
echo   - Details CSV : reports\geo_validation\geo_benchmark_17k_baseline_details.csv
echo   - Log fichier : %LOG_DIR%\run_geo_benchmark_17k.log
echo ============================================================
echo.
pause

ENDLOCAL
