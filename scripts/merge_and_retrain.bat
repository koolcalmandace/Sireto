@echo off
REM ============================================================
REM merge_and_retrain.bat
REM Script d'execution locale pour la fusion CRM + reentainement V8b
REM Branche : feature/geo-resolution-and-crm-expansion
REM Date    : 2026-08-20
REM ============================================================
REM
REM PREREQUIS :
REM   - Python 3.14+ avec les dependances du projet installees
REM   - data/crm_ok_gt_increment_20260817.csv present
REM   - data/candidates_v7_all/ present (partitions SIRENE)
REM   - data/StockEtablissement_utf8.parquet present (pour siren_to_geo)
REM   - models/xgbranker_fast_*.json  (Stage 1 actuel)
REM   - models/xgb_decider_*.json     (Stage 2 actuel)
REM
REM EXECUTION : Lancez ce fichier depuis la racine du projet
REM   Clic-droit > Executer en tant qu administrateur
REM   ou depuis PowerShell : .\scripts\merge_and_retrain.bat
REM ============================================================

SETLOCAL

REM Always cd to project root (parent folder of scripts\)
cd /d "%~dp0\.."

SET PYTHON=py
SET LOG_DIR=logs\retraining_v8b_merged
IF NOT EXIST %LOG_DIR% MKDIR %LOG_DIR%

echo.
echo ============================================================
echo ETAPE 1/6 : FUSION CRM  (crm_ok_gt_merged_v1.csv)
echo ============================================================
%PYTHON% scripts\merge_crm_increment.py ^
    --orig data\crm_ok_gt.csv ^
    --increment data\crm_ok_gt_increment_20260817.csv ^
    --output data\crm_ok_gt_merged_v1.csv
IF ERRORLEVEL 1 (echo ECHEC ETAPE 1 & EXIT /B 1)
echo ETAPE 1 OK — data\crm_ok_gt_merged_v1.csv cree

echo.
echo ============================================================
echo ETAPE 2/6 : CONSTRUCTION siren_to_geo.parquet (--geo-only)
echo ============================================================
%PYTHON% scripts\build_siren_global_index.py ^
    --partitions-dir data\candidates_v7_all ^
    --output-dir data\siren_index ^
    --geo-only ^
    2>&1 | tee %LOG_DIR%\step2_siren_to_geo.log
IF ERRORLEVEL 1 (echo ECHEC ETAPE 2 & EXIT /B 1)
echo ETAPE 2 OK — data\siren_index\siren_to_geo.parquet cree

echo.
echo ============================================================
echo ETAPE 3/6 : GENERATION DES SAMPLES RANKER V8b-merged
echo ============================================================
%PYTHON% scripts\generate_training_samples_v5fast.py ^
    --crm data\crm_ok_gt_merged_v1.csv ^
    --partitions-dir data\candidates_v7_all ^
    --output data\samples_v8b_merged_ranker.parquet ^
    --mode ranker ^
    --enable-siren-expansion ^
    --siren-index data\siren_index ^
    2>&1 | tee %LOG_DIR%\step3_gen_ranker.log
IF ERRORLEVEL 1 (echo ECHEC ETAPE 3 & EXIT /B 1)
echo ETAPE 3 OK — samples ranker generes

echo.
echo ============================================================
echo ETAPE 4/6 : GENERATION DES SAMPLES DECIDER V8b-merged
echo ============================================================
%PYTHON% scripts\generate_training_samples_v5fast.py ^
    --crm data\crm_ok_gt_merged_v1.csv ^
    --partitions-dir data\candidates_v7_all ^
    --output data\samples_v8b_merged_decider.parquet ^
    --mode decider ^
    --enable-siren-expansion ^
    --siren-index data\siren_index ^
    2>&1 | tee %LOG_DIR%\step4_gen_decider.log
IF ERRORLEVEL 1 (echo ECHEC ETAPE 4 & EXIT /B 1)
echo ETAPE 4 OK — samples decider generes

echo.
echo ============================================================
echo ETAPE 5/6 : REENTRAINEMENT STAGE 1 (Fast Ranker)
echo ============================================================
%PYTHON% scripts\train_xgb_ranker.py ^
    --samples data\samples_v8b_merged_ranker.parquet ^
    --output-dir models ^
    --tag v8b_merged ^
    2>&1 | tee %LOG_DIR%\step5_train_ranker.log
IF ERRORLEVEL 1 (echo ECHEC ETAPE 5 & EXIT /B 1)
echo ETAPE 5 OK — Stage 1 Fast Ranker entraine

echo.
echo ============================================================
echo ETAPE 6/6 : REENTRAINEMENT STAGE 2 (Semantic Decider)
echo ============================================================
%PYTHON% scripts\train_xgb_decider.py ^
    --samples data\samples_v8b_merged_decider.parquet ^
    --output-dir models ^
    --tag v8b_merged ^
    2>&1 | tee %LOG_DIR%\step6_train_decider.log
IF ERRORLEVEL 1 (echo ECHEC ETAPE 6 & EXIT /B 1)
echo ETAPE 6 OK — Stage 2 Semantic Decider entraine

echo.
echo ============================================================
echo NOTE : ETAPE 7 (Recalibration Stage 3 Risk Model)
echo ============================================================
echo Apres avoir valide les performances Stage 1+2 sur le dev set,
echo lancer manuellement :
echo.
echo   py scripts\train_routing_risk_model.py ^
echo       --eval-results reports\benchmark_v8b_merged_topk.csv ^
echo       --output models\routing_risk_model_v8b_merged.pkl
echo.
echo Puis mettre a jour le seuil AUTO dans handover.md.
echo ============================================================
echo.
echo TOUTES LES ETAPES TERMINEES AVEC SUCCES.
echo Resultats dans : %LOG_DIR%\
echo Dataset fusionne : data\crm_ok_gt_merged_v1.csv

ENDLOCAL
