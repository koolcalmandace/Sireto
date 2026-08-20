# Plan d'Optimisation des Poids & Réentraînement V8b (En attente / Standby)

**Statut** : EN ATTENTE (STANDBY)  
**Date** : 20 Août 2026  
**Objectif** : Maximiser le taux d'Auto Match (> 75-78%) tout en maintenant une précision réelle $\ge 99.80\%$ sur le jeu complet de 17 054 requêtes CRM.

---

## 1. Contexte & Problématique

Sur le jeu de données de référence (17 054 requêtes CRM), l'analyse des poids et fonctions de perte actuels a mis en évidence plusieurs leviers d'amélioration :

1. **Stage 1 (Ranker)** : Actuellement en `rank:pairwise` ; le passage en `rank:ndcg` (LambdaMART) permet une meilleure concentration sur le Recall@50.
2. **Stage 2 (Decider - Poids de classe)** : `scale_pos_weight = 100` sur-pondère artificiellement les positifs par rapport au ratio 1:50 réel, ce qui tasse les écarts de score (`score_gap`). Un réajustement à `50` (ou passage en objectif listwise) restaure le contraste discriminant.
3. **Stage 3 (Risk Model - Seuil)** : Le seuil `0.835` a été calibré sur la distribution V7. L'introduction de l'expansion SIREN (V8b) et des 7 nouvelles features d'interaction nécessite une recalibration empirique par régression isotonique.

---

## 2. Propositions d'Optimisation des Poids

### A. Stage 1 - Fast Ranker
- **Paramètre actuel** : `objective: "rank:pairwise"`, `eta: 0.1`, `max_depth: 6`
- **Proposition** : Tester `objective: "rank:ndcg"`, `eval_metric: ["ndcg@50", "map@50"]` pour optimiser directement le rappel sur le top 50 sans pénaliser les inversions en bas de liste.

### B. Stage 2 - Semantic Decider
- **Paramètre actuel** : `objective: "binary:logistic"`, `scale_pos_weight: 100`, `learning_rate: 0.05`, `max_depth: 7`, `min_child_weight: 3`
- **Proposition** :
  - Option 1 (Classification calibrée) : Réduire `scale_pos_weight` à `50` pour refléter exactement le ratio d'échantillonnage 1:50.
  - Option 2 (Ranking listwise) : Évaluer `objective: "rank:ndcg"` sur les top-50 candidats du Stage 1.

### C. Stage 3 - Risk Model & Seuil de Routing
- **Paramètre actuel** : Seuil figé à `0.835`
- **Proposition** : Ré-estimer le seuil optimal sur l'ensemble de dev/test V8b pour cibler précisément $FP\_rate \le 0.16\%$.

---

## 3. Séquence d'Exécution Locale Prévue (Standby)

Dès activation, les étapes seront exécutées localement via scripts `.bat` :

### Étape 1 : Construction de l'index de géolocalisation SIREN (si manquant)
```bash
python scripts/build_siren_global_index.py --geo-only --etab-path data/StockEtablissement_utf8.parquet --ul-path data/StockUniteLegale_utf8.parquet --output-dir data/siren_index
```

### Étape 2 : Génération des samples V8b (Ranker & Decider)
```bash
python scripts/generate_training_samples_v5fast.py --mode ranker --enable-siren-expansion --siren-to-geo data/siren_index/siren_to_geo.parquet --output data/samples_v8b_ranker.parquet
python scripts/generate_training_samples_v5fast.py --mode decider --enable-siren-expansion --siren-to-geo data/siren_index/siren_to_geo.parquet --output data/samples_v8b_decider.parquet
```

### Étape 3 : Entraînement Stage 1 & Stage 2
```bash
python scripts/train_xgb_ranker.py --samples-path data/samples_v8b_ranker.parquet --output-dir models/
python scripts/train_xgb_decider.py --samples-path data/samples_v8b_decider.parquet --output-dir models/
```

### Étape 4 : Inférence, Recalibration Stage 3 & Évaluation
```bash
python scripts/train_routing_risk_model.py --dataset-path reports/benchmark_v8b_topk.csv --output-dir models/ --target siren --calibration isotonic
python scripts/evaluate_routing.py --input-path reports/routed_v8b.csv --gt-path data/crm_ok_gt.csv
```

---

## 4. Marqueurs d'Évaluation Attendus (Post-Run)
1. **Retrieval elimination** (Couverture GT dans le pool top-k).
2. **Répartition du Routing** : Auto Match Rate (%), Review Rate (%), No Match Rate (%).
3. **Automatch Precision** ($\ge 99.80\%$).
4. **Précision SIREN & SIRET** séparées.
