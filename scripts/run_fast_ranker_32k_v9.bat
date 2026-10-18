@echo off
REM ============================================================
REM run_fast_ranker_32k_v9.bat
REM Evaluation complete du Fast Ranker V9 sur 32K National
REM avec detection continue des cas repeches vs baseline V8 Retention V2.
REM Branche : feature/fast-ranker-v9-ndcg50-retraining
REM ============================================================

SETLOCAL
cd /d "%~dp0\.."

SET PYTHON=python
SET LOG_DIR=logs
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%
IF NOT EXIST reports\fast_ranker_32k MKDIR reports\fast_ranker_32k

echo.
echo ============================================================
echo EXECUTION DU BENCHMARK NATIONAL FAST RANKER 32K (V9 RETENTION V2)
echo [SECURITE ACTIVE] Checkpointing atomique + Anti-Veille + Suivi Repechage
echo   - Tag dedie : 32k_v9_retention_v2
echo   - Modele 47 features : models\xgbranker_fast_v9.json
echo   - Baseline reference : reports\fast_ranker_32k\fast_ranker_32k_v8_retention_v2_details.csv
echo   - Mode continu : --no-pause-on-lock (zero veille / zero interruption)
echo   - Reprise automatique : --resume actif par defaut
echo ============================================================
%PYTHON% scripts\run_v9_32k_sentinel.py

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] L evaluation a rencontre une erreur.
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Evaluation 32K V9 terminee avec succes !
echo Rapports generes :
echo   - Resume JSON  : reports\fast_ranker_32k\fast_ranker_32k_v9_retention_v2_summary.json
echo   - Details CSV  : reports\fast_ranker_32k\fast_ranker_32k_v9_retention_v2_details.csv
echo   - Repeches CSV : reports\fast_ranker_32k\fast_ranker_32k_v9_retention_v2_recovered_cases.csv
echo   - Log fichier  : %LOG_DIR%\run_fast_ranker_32k_v9.log
echo ============================================================

ENDLOCAL
