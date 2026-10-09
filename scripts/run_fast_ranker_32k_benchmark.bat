@echo off
REM ============================================================
REM run_fast_ranker_32k_benchmark.bat
REM Evaluation du Fast Ranker (Stage 1) sur le dataset national 32K
REM avec retention Siege Social et Sidecar et 38 features rapides.
REM Branche : experiment/fast-ranker-32k-retention
REM ============================================================

SETLOCAL
cd /d "%~dp0\.."

SET PYTHON=python
SET LOG_DIR=logs
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%
IF NOT EXIST reports\fast_ranker_32k MKDIR reports\fast_ranker_32k

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK FAST RANKER 32K (STAGE 1 UNIFIE BAN)
echo [SECURITE ACTIVE] Checkpointing atomique + Mode Anti-Veille
echo   - Tag dedie : 32k_v8_unified_ban
echo   - Modele 42 features : models\xgbranker_fast_v8.json
echo   - Reprise automatique en cas d interruption (--resume par defaut)
echo   - Retention maximisee K=500 vers K=50 (Siege + Sidecar + BAN GPS)
echo ============================================================
%PYTHON% scripts\evaluate_fast_ranker_32k.py ^
    --crm data\crm_ok_gt_merged_v1.csv ^
    --partitions-dir data\candidates_v8_national ^
    --output-dir reports\fast_ranker_32k ^
    --tag 32k_v8_unified_ban ^
    --model-path models\xgbranker_fast_v8.json ^
    --prefilter-k 500 ^
    --top-k 50 ^
    --cache-size 40 ^
    --max-ram-gb 3.5 ^
    --checkpoint-interval 250 ^
    --no-pause-on-lock ^
    --log-file %LOG_DIR%\run_fast_ranker_32k_unified_ban.log

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] L evaluation a rencontre une erreur.
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Evaluation terminee avec succes !
echo Rapports generes :
echo   - Resume JSON : reports\fast_ranker_32k\fast_ranker_32k_v8_unified_ban_summary.json
echo   - Details CSV : reports\fast_ranker_32k\fast_ranker_32k_v8_unified_ban_details.csv
echo   - Log fichier : %LOG_DIR%\run_fast_ranker_32k_unified_ban.log
echo ============================================================

ENDLOCAL
