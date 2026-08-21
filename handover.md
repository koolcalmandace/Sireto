# SIRETO Handover - 21 Aout 2026

## Etat des lieux
La feuille de route d'execution progressive est active :
1. **Etape 1 (Faite)** : Câblage de la resolution geo (`load_with_geo_resolution`) dans `retrieval.py` et alignement sans skew dans `infer.py`.
2. **Etape 2 (En cours)** : Benchmark et validation empirique sur les 17 054 cas CRM baseline pour certifier l'absence de regression.
3. **Etape 3 (A venir)** : Evaluation de l'incrément 15 516.
4. **Etape 4 (A venir)** : Fusion controlee des datasets (`crm_ok_gt_merged_v1.csv` : 32 570 lignes).
5. **Etape 5 (A venir)** : Audit approfondi de l'architecture (Step 2 Retrieval -> Step 3 Features -> Stages ML) avant le reentrainement des poids.

**Branche active** : `feature/geo-resolution-and-crm-expansion` (depuis `main`)

## Actions terminees (fenetre recente)
- **Amendement rules.md (Project North Star)** : Ajout de la Regle 7 pour la centralisation, la tracabilite et le versionnage systematique des livrables (planning, resultats, documents, programmes console). *(commit GitHub: `ed4daeb`)*
- **Etape 1 : Unification Geo-Resolution sans Skew** : Branchement de `store.load_with_geo_resolution()` directement dans `build_candidate_pool()` de `src/xgb_matcher/retrieval.py` et nettoyage de la delegation dans `src/xgb_matcher/infer.py`. *(commit GitHub: `fc4396b`)*
- **Scripts et Outils d'Evaluation Geo (Etape 2 & 3)** : Creation de `scripts/evaluate_geo_resolution.py` (analyse detaillee base recall, prefilter top-K, loss reasons, stratification `loc_match_type` et `sirene_etat`), avec scripts batch `scripts/run_geo_benchmark_17k.bat` et `scripts/run_geo_benchmark_increment.bat` (avec resolution automatique du dossier racine `cd /d "%~dp0\.."`). *(commit GitHub: `2d02cf8`)*
- **Guide d'Execution Utilisateur** : `docs/plans/step2_geo_validation_execution_guide.md`. *(commit GitHub: `fc4396b`)*
- **Chantier 1 : Resolution Geo Propre** : ajout `load_with_geo_resolution()` + `_discover_insee_codes_from_cp()` dans `partitioned_store.py` ; hierarchie stricte : INSEE -> CP child discovery -> LOG_EMPTY. *(commit GitHub: `c754b97`)*
- **Chantier 2a : Audit qualite increment CRM 20260817** : 15 516 cas UNSEEN valides, 0 ambigu, 0% chevauchement SIREN, verdict SATISFAISANT. *(commit GitHub: `c754b97`)*
- **Chantier 2b : Script de fusion controlee** : `scripts/merge_crm_increment.py` produit `data/crm_ok_gt_merged_v1.csv` (32 570 lignes, 0 doublon). *(commit GitHub: `c754b97`)*
- **V8b SIREN expansion (V7 + local + cross-partition)**: ajout Step 5 d'expansion apres prefilter, feature flag, cap pool dedie et telemetrie d'expansion. *(commit GitHub: `9c0e806`, `f1fbbb8`, `c961371`)*

## Fichiers modifies recemment
- `src/xgb_matcher/retrieval.py`
- `src/xgb_matcher/infer.py`
- `scripts/evaluate_geo_resolution.py` (nouveau)
- `scripts/run_geo_benchmark_17k.bat` (nouveau)
- `scripts/run_geo_benchmark_increment.bat` (nouveau)
- `docs/plans/step2_geo_validation_execution_guide.md` (nouveau)

## Prochaines etapes
1. **Lancement du benchmark 17k par l'utilisateur** via `.\scripts\run_geo_benchmark_17k.bat`.
2. **Analyse des resultats et validation de non-regression** sur `reports/geo_validation/geo_benchmark_17k_baseline_summary.json` (cibles : Base Recall $\ge 99.1\%$, Prefilter@500 $\ge 98.8\%$).
3. **Execution du benchmark sur l'increment 15 516** via `.\scripts\run_geo_benchmark_increment.bat`.
4. **Execution de la fusion** via `python scripts/merge_crm_increment.py` (production de `data/crm_ok_gt_merged_v1.csv`).
5. **Poursuite de l'audit architectural pas-a-pas** (Etape 2 Retrieval TF-IDF/Rescue -> Etape 3 Feature Engineering -> Stages ML).
6. **Reentrainement et recalibration des poids** (Standby).

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*

