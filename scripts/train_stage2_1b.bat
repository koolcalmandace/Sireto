@echo off
REM =========================================================================
REM SIRETO V7 — ENTRAINEMENT STAGE 2 ARCHITECTURE 1B (72 FEATURES + 32K SPLIT)
REM   - Specialist 1 : XGBRanker (rank:pairwise, ndcg@1)
REM   - Specialist 2 : XGBClassifier (binary:logistic) + SigmoidCalibrator
REM   - Split        : 70% Train / 15% Dev / 15% Locked Test (GroupKFold SIREN)
REM   - Resilience   : Atomic Checkpoints + PowerGuard (--resume)
REM =========================================================================
echo ===============================================================================
echo SIRETO V7 - ENTRAINEMENT STAGE 2 (1B: RANKER + PLATT CONFIDENCE SCORER)
echo ===============================================================================
echo.
"C:\Users\Kabouassi\AppData\Local\Programs\Python\Python314\python.exe" scripts\train_unified_xgb_decider.py --sample-size 0 --calibration sigmoid --resume
pause
