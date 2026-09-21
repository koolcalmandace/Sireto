@echo off
REM ============================================================
REM run_geo_benchmark_32k.bat
REM Evaluation finale de la resolution Geo & Recall sur l'ensemble fusionne 32 570
REM Branche : feature/retrieval-public-communal-whitelisting
REM ============================================================

SETLOCAL
cd /d "%~dp0\.."

SET PYTHON=py
SET LOG_DIR=logs\geo_validation
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK GEO-RESOLUTION (32 570 CRM Merged)
echo [SECURITE ACTIVE] Checkpointing atomique + Moniteur Veille PowerGuard
echo   - Reprise automatique en cas d'interruption (--resume par defaut)
echo   - Pause propre lors de la veille / fermeture capot (0%% CPU)
echo ============================================================
%PYTHON% scripts\evaluate_geo_resolution.py ^
    --crm data\crm_ok_gt_merged_v1.csv ^
    --partitions-dir data\candidates_v8_national ^
    --output-dir reports\geo_validation ^
    --tag 32k_merged_v8 ^
    --prefilter-k 500 ^
    --cache-size 150 ^
    --max-ram-gb 4.0 ^
    --checkpoint-interval 100 ^
    --log-file %LOG_DIR%\run_geo_benchmark_32k.log

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] L'evaluation a rencontre une erreur.
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Evaluation terminee avec succes !
echo Rapports generes :
echo   - Resume JSON : reports\geo_validation\geo_benchmark_32k_merged_v8_summary.json
echo   - Details CSV : reports\geo_validation\geo_benchmark_32k_merged_v8_details.csv
echo   - Log fichier : %LOG_DIR%\run_geo_benchmark_32k.log
echo ============================================================

ENDLOCAL
