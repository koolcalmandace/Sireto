# Guide d'Exécution & Protocole de Validation : Étape 2 (Benchmark Geo-Resolution 17k)

**Date** : 21 Août 2026  
**Branche** : `feature/geo-resolution-and-crm-expansion`  
**Statut** : PRÊT POUR EXÉCUTION LOCALE PAR L'UTILISATEUR  

---

## 1. Contexte & Alignement Technique (Étape 1 Réalisée)

Avant d'exécuter ce benchmark, le câblage de la **résolution géographique stricte** (`load_with_geo_resolution`) a été unifié sur l'ensemble de la base de code pour garantir un **zéro skew** entre train, serve et benchmark :
1. [`src/xgb_matcher/retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py) : `build_candidate_pool()` appelle directement `store.load_with_geo_resolution()` (avec gestion de la hiérarchie INSEE $\rightarrow$ découverte des enfants INSEE par le CP $\rightarrow$ pool vide).
2. [`src/xgb_matcher/infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py) : `_build_candidate_pool()` délègue directement à `build_candidate_pool()` sans duplication de code.

---

## 2. Commandes d'Exécution de l'Étape 2 (Sur votre machine)

### Option A : Via le script Batch (Recommandé)
Depuis la racine du projet, double-cliquez ou lancez dans le terminal :
```bat
.\scripts\run_geo_benchmark_17k.bat
```

### Option B : Directement en Python / PowerShell
```powershell
py scripts/evaluate_geo_resolution.py --crm data/crm_ok_gt.csv --partitions-dir data/candidates_v7_all --output-dir reports/geo_validation --tag 17k_baseline --prefilter-k 500
```

---

## 3. Fichiers et Résultats Générés (Attendus)

Après l'exécution, les artefacts suivants seront créés dans `reports/geo_validation/` :
- `reports/geo_validation/geo_benchmark_17k_baseline_summary.json` : Résumé consolidé des taux de recall globaux et par segment (`loc_match_type`, `sirene_etat`).
- `reports/geo_validation/geo_benchmark_17k_baseline_details.csv` : Audit détaillé ligne par ligne avec statut GT, taille des pools et motif d'échec (`loss_reason`).
- `logs/geo_validation/run_geo_benchmark_17k.log` : Logs complets de la session.

---

## 4. Métriques Cibles & Seuils de Non-Régression

| Métrique | Valeur Cible (Baseline SOTA) | Seuil d'Alerte Régression |
| :--- | :---: | :---: |
| **Total Requêtes Évaluées** | **17 054** | - |
| **Recall Base Pool (Présence GT)** | **$\ge 99.1\%$** | $< 99.0\%$ |
| **Recall Top-500 Prefilter (TF-IDF + Rescue)** | **$\ge 98.8\%$** | $< 98.5\%$ |
| **Recall CP-Only (279 cas sans INSEE)** | **$\ge 92.0\%$** | $< 88.0\%$ |
| **Établissements Fermés (3 213 cas)** | Conservés dans le pool ($\ge 98.5\%$) | $< 98.0\%$ |

---

## 5. Prochaines Actions Post-Exécution

Dès que vous aurez terminé le run :
1. Vous pourrez partager les logs console ou le fichier JSON `reports/geo_validation/geo_benchmark_17k_baseline_summary.json`.
2. Nous lancerons immédiatement le diagnostic comparatif pour valider l'absence de régression.
3. Si le résultat est validé, nous lancerons l'évaluation sur l'incrément de 15 516 (`scripts/run_geo_benchmark_increment.bat`), puis la fusion (`merge_crm_increment.py`).
