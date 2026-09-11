@echo off
REM ============================================================
REM run_build_national_partitions.bat
REM Generation haute performance du store partitionne SIRENE National V8
REM Source  : data\source_sirene_20260901\ (Snapshot Septembre 2026)
REM Cible   : data\candidates_v8_national\
REM ============================================================

SETLOCAL
cd /d "%~dp0\.."

SET PYTHON=py
SET LOG_DIR=logs\sirene_national
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%

echo.
echo ============================================================
echo LANCEMENT DE LA GENERATION NATIONALE SIRENE V8
echo ============================================================
echo Source Etab : data\source_sirene_20260901\StockEtablissement_utf8_20260901.parquet
echo Source UL   : data\source_sirene_20260901\StockUniteLegale_utf8_20260901.parquet
echo Cible       : data\candidates_v8_national\
echo.

%PYTHON% scripts\build_candidate_partitions_national.py ^
    --etab-path data\source_sirene_20260901\StockEtablissement_utf8_20260901.parquet ^
    --ul-path data\source_sirene_20260901\StockUniteLegale_utf8_20260901.parquet ^
    --output-dir data\candidates_v8_national ^
    --threads 8

IF ERRORLEVEL 1 (
    echo.
    echo [ERREUR] La generation a echoue.
    EXIT /B 1
)

echo.
echo ============================================================
echo [SUCCES] Generation terminee avec succes !
echo Partitions crees dans : data\candidates_v8_national\
echo Log disponible dans   : %LOG_DIR%\build_national_partitions.log
echo ============================================================

ENDLOCAL
