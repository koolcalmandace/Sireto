@echo off
REM ============================================================
REM run_merge_crm.bat
REM Execution securisee de la fusion CRM (Etape 4)
REM Bascule automatiquement vers la racine du projet Sireto-dev
REM ============================================================

SETLOCAL

REM Toujours se positionner a la racine du projet Sireto-dev
cd /d "%~dp0\.."

SET PYTHON=py

echo.
echo ============================================================
echo FUSION CONTROLEE DU DATASET CRM (ETAPE 4)
echo   - Baseline : data\crm_ok_gt.csv (17 054 requetes)
echo   - Increment: data\crm_ok_gt_increment_20260817.csv (15 516 UNSEEN)
echo   - Cible    : data\crm_ok_gt_merged_v1.csv (32 570 requetes)
echo ============================================================
echo Repertoire de travail actif : %CD%
echo.

%PYTHON% scripts\merge_crm_increment.py

IF ERRORLEVEL 1 (
    echo.
    echo [ECHEC] La fusion a rencontre une erreur.
    echo.
    pause
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Fusion terminee avec succes !
echo Fichier genere : data\crm_ok_gt_merged_v1.csv
echo ============================================================
echo.
pause

ENDLOCAL
