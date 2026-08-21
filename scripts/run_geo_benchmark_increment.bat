@echo off
REM ============================================================
REM run_geo_benchmark_increment.bat
REM Evaluation de la resolution Geo & Recall sur l'increment 15 516
REM Branche : feature/geo-resolution-and-crm-expansion
REM Date    : 2026-08-21
REM ============================================================

SETLOCAL

REM Always cd to project root (parent folder of scripts\)
cd /d "%~dp0\.."

SET PYTHON=py
SET LOG_DIR=logs\geo_validation
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK GEO-RESOLUTION (15 516 CRM Increment)
echo ============================================================
%PYTHON% scripts\evaluate_geo_resolution.py ^
    --crm data\crm_ok_gt_increment_20260817.csv ^
    --partitions-dir data\candidates_v7_all ^
    --output-dir reports\geo_validation ^
    --tag 15k_increment ^
    --prefilter-k 500 ^
    --log-file %LOG_DIR%\run_geo_benchmark_increment.log

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
echo   - Resume JSON : reports\geo_validation\geo_benchmark_15k_increment_summary.json
echo   - Details CSV : reports\geo_validation\geo_benchmark_15k_increment_details.csv
echo   - Log fichier : %LOG_DIR%\run_geo_benchmark_increment.log
echo ============================================================
echo.
pause

ENDLOCAL
