REM ============================================================
REM  TYPE    : SPECIAL VARIANT
REM  VERSION : V0 — Model Training
REM  STATUS  : Trains XGBoost Ranker, Decider, and Risk models from scratch.
REM ============================================================

@echo off
title Sireto - Train XGBoost ML Models From Scratch
echo ==========================================================
echo Starting XGBoost ML Model Training Pipeline...
echo Database: data/harvest_full.sqlite
echo CRM Source: data/crm_ok_gt.csv
echo ==========================================================
cd /d "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"
set PYTHONPATH=src;.

:: Enable semantic features
set XGB_SEMANTIC_ENABLED=1

:: Enable parallel workers (6 out of 8 logical processors)
set XGB_SAMPLE_WORKERS=6

echo.
if exist "data\candidates_v7_all\cp" (
    echo [INFO] Partitioned candidates folder already exists.
    echo [STEP 1/5] Skipping partition building.
) else (
    echo [STEP 1/5] Building partitioned candidates from SQLite database...
    .\venv\Scripts\python.exe scratch/build_partitions_from_sqlite.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Step 1 failed! Partition building aborted.
        pause
        exit /b 1
    )
)

echo.
if exist "data\samples_v5_ranker.parquet" (
    echo [INFO] Ranker samples already exist.
    echo [STEP 2/5] Skipping ranker sample generation.
) else (
    echo [STEP 2/5] Generating training samples for Ranker model...
    .\venv\Scripts\python.exe scripts/generate_training_samples_v5fast.py --mode=ranker --output data/samples_v5_ranker.parquet
    if errorlevel 1 (
        echo.
        echo [ERROR] Step 2 failed! Ranker sample generation aborted.
        pause
        exit /b 1
    )
)

echo.
dir /b models\xgbranker_*.json >nul 2>nul
if not errorlevel 1 (
    echo [INFO] Ranker model files already exist.
    echo [STEP 3/5] Skipping ranker model training.
) else (
    echo [STEP 3/5] Training Ranker model [Stage 1]...
    .\venv\Scripts\python.exe scripts/train_xgb_ranker.py --samples data/samples_v5_ranker.parquet
    if errorlevel 1 (
        echo.
        echo [ERROR] Step 3 failed! Ranker training aborted.
        pause
        exit /b 1
    )
)

echo.
if exist "data\samples_v5_decider.parquet" (
    echo [INFO] Decider samples already exist.
    echo [STEP 4/5] Skipping decider sample generation.
) else (
    echo [STEP 4/5] Generating training samples for Decider model [using trained Ranker]...
    .\venv\Scripts\python.exe scripts/generate_training_samples_v5fast.py --mode=decider --output data/samples_v5_decider.parquet
    if errorlevel 1 (
        echo.
        echo [ERROR] Step 4 failed! Decider sample generation aborted.
        pause
        exit /b 1
    )
)

echo.
.\venv\Scripts\python.exe -c "import json, glob; exit(0 if any('decider_model' in json.load(open(f)) for f in glob.glob('models/xgb_two_stage_meta_*.json')) else 1)"
if not errorlevel 1 (
    echo [INFO] Decider model and metadata already exist.
    echo [STEP 5/5] Skipping decider model training.
) else (
    echo [STEP 5/5] Training Decider model [Stage 2] and calibrating...
    .\venv\Scripts\python.exe scripts/train_xgb_decider.py --samples data/samples_v5_decider.parquet
    if errorlevel 1 (
        echo.
        echo [ERROR] Step 5 failed! Decider training aborted.
        pause
        exit /b 1
    )
)

echo.
echo ==========================================================
echo [SUCCESS] All 5 training steps completed successfully!
echo XGBoost models are saved in the "models/" folder.
echo You can now run the optimized ML matching pipeline using:
echo Run_ML_Full_Opt.bat
echo ==========================================================
pause
