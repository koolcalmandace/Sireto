@echo off
title Sireto - Train XGBoost ML Models Version 4.2 (Expanded Confirmed)
echo ==========================================================
echo Starting XGBoost ML Model Training Pipeline Version 4.2...
echo Database: data/harvest_full.sqlite
echo CRM Source: data/crm_ok_gt_confirmed_expanded.csv (6.7K Confirmed)
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

:: Enable parallel workers (6 out of 8 logical processors)
set XGB_SAMPLE_WORKERS=6

echo.
echo [STEP 1/5] Backing up old samples...
if exist "data\samples_v5_ranker.parquet" (
    echo Moving old ranker samples to data\samples_v5_ranker_v40_backup.parquet...
    move /y "data\samples_v5_ranker.parquet" "data\samples_v5_ranker_v40_backup.parquet" >nul
)
if exist "data\samples_v5_decider.parquet" (
    echo Moving old decider samples to data\samples_v5_decider_v40_backup.parquet...
    move /y "data\samples_v5_decider.parquet" "data\samples_v5_decider_v40_backup.parquet" >nul
)

echo.
echo [STEP 2/5] Generating training samples for Ranker model (Stage 1)...
set XGB_SEMANTIC_ENABLED=0
.\venv\Scripts\python.exe scripts/generate_training_samples_v5fast.py --mode=ranker --training-csv data/crm_ok_gt_confirmed_expanded.csv --output data/samples_v5_ranker.parquet
if errorlevel 1 (
    echo.
    echo [ERROR] Step 2 failed! Ranker sample generation aborted.
    pause
    exit /b 1
)

echo.
echo [STEP 3/5] Training Ranker model (Stage 1)...
.\venv\Scripts\python.exe scripts/train_xgb_ranker.py --samples data/samples_v5_ranker.parquet
if errorlevel 1 (
    echo.
    echo [ERROR] Step 3 failed! Ranker training aborted.
    pause
    exit /b 1
)

echo.
echo [STEP 4/5] Generating training samples for Decider model (Stage 2)...
set XGB_SEMANTIC_ENABLED=1
.\venv\Scripts\python.exe scripts/generate_training_samples_v5fast.py --mode=decider --training-csv data/crm_ok_gt_confirmed_expanded.csv --output data/samples_v5_decider.parquet
if errorlevel 1 (
    echo.
    echo [ERROR] Step 4 failed! Decider sample generation aborted.
    pause
    exit /b 1
)

echo.
echo [STEP 5/5] Training Decider model (Stage 2) and calibrating...
.\venv\Scripts\python.exe scripts/train_xgb_decider.py --samples data/samples_v5_decider.parquet
if errorlevel 1 (
    echo.
    echo [ERROR] Step 5 failed! Decider training aborted.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo [SUCCESS] All 5 training steps completed successfully!
echo XGBoost Version 4.2 models are saved in the "models/" folder.
echo ==========================================================
pause
