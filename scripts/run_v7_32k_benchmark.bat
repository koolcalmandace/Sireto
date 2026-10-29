@echo off
REM =========================================================================
REM SIRETO V7 — LANCEMENT OFFICIEL DU BENCHMARK NATIONAL COMPLET 32K
REM Architecture 100%% locale et deterministe :
REM   - Stage 1 : Fast Ranker V9 NDCG@50 (47 features)
REM   - Stage 2 : Unified Decider 67 features (calibre isotonique)
REM   - Tie-Breaker Engine : Departagement deterministe intra-SIREN et inter-SIREN
REM   - Stage 3 : Routing Risk Model (calibrateur isotonique, seuil 0.85)
REM   - Power Guard : Pause sur capot ferme / verrouillage session, reprise fluide
REM =========================================================================
echo ===============================================================================
echo SIRETO V7 - LANCEMENT DU BENCHMARK NATIONAL 32K (32 570 REQUETES)
echo ===============================================================================
echo.
python scripts\run_v7_32k_sentinel.py --resume
pause
