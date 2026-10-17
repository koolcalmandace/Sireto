@echo off
REM ============================================================
REM run_fast_ranker_32k_retention_v2.bat
REM Evaluation complete du Fast Ranker (Stage 1) sur 32K National
REM avec detection continue des cas repeches vs baseline.
REM Branche : feature/retention-sanitization-v2
REM ============================================================

SETLOCAL
cd /d "%~dp0\.."

SET PYTHON=python
SET LOG_DIR=logs
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%
IF NOT EXIST reports\fast_ranker_32k MKDIR reports\fast_ranker_32k

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK NATIONAL FAST RANKER 32K (RETENTION V2)
echo [SECURITE ACTIVE] Checkpointing atomique + Anti-Veille + Suivi Repechage
echo   - Tag dedie : 32k_v8_retention_v2
echo   - Modele 42 features : models\xgbranker_fast_v8.json
echo   - Baseline reference : reports\fast_ranker_32k\fast_ranker_32k_v8_unified_ban_details.csv
echo   - Mode continu : --no-pause-on-lock (zero veille / zero interruption)
echo   - Reprise automatique : --resume actif par defaut
echo ============================================================
%PYTHON% scripts\evaluate_fast_ranker_32k.py ^
    --crm data\crm_ok_gt_merged_v1.csv ^
    --partitions-dir data\candidates_v8_national ^
    --output-dir reports\fast_ranker_32k ^
    --tag 32k_v8_retention_v2 ^
    --model-path models\xgbranker_fast_v8.json ^
    --baseline-details reports\fast_ranker_32k\fast_ranker_32k_v8_unified_ban_details.csv ^
    --prefilter-k 500 ^
    --top-k 50 ^
    --cache-size 40 ^
    --max-ram-gb 3.5 ^
    --checkpoint-interval 250 ^
    --no-pause-on-lock ^
    --log-file %LOG_DIR%\run_fast_ranker_32k_retention_v2.log

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] L evaluation a rencontre une erreur.
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Evaluation 32K terminee avec succes !
echo Rapports generes :
echo   - Resume JSON  : reports\fast_ranker_32k\fast_ranker_32k_v8_retention_v2_summary.json
echo   - Details CSV  : reports\fast_ranker_32k\fast_ranker_32k_v8_retention_v2_details.csv
echo   - Repeches CSV : reports\fast_ranker_32k\fast_ranker_32k_v8_retention_v2_recovered_cases.csv
echo   - Log fichier  : %LOG_DIR%\run_fast_ranker_32k_retention_v2.log
echo ============================================================

ENDLOCAL
