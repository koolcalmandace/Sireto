# Phase 1 : Transition Pipe V6 (LLM) vers Pipe V7 (100% Déterministe)

**Période** : Janvier - Février 2026  
**Objectif Majeur** : Remplacer l'arbitrage par LLMs (Ollama) par une architecture 100% déterministe haute performance à base de XGBoost 3-Stages et fallback Google Places.

---

## 1. Contexte & Motivations du Pivot
* **Limites de la V6** : Coûts GPU, latence de plusieurs secondes par requête, instabilité non-déterministe des réponses LLM.
* **Nouvelle Vision V7** : Pipeline ML pur (`XGBRanker` $\rightarrow$ `XGBClassifier` $\rightarrow$ `Risk Model`) atteignant **74.5% d'Auto-Match à 99.84% de précision** sur 2 512 requêtes Test.

---

## 2. Décisions & Réalisations Majeures
1. **Multi-Blocking & Stockage Hive Parquet** :
   - Partitionnement par code INSEE et code postal (`data/candidates_v7_all/`).
   - Manifeste d'indexation $O(1)$ (`insee_counts.parquet`) et politique méga-communes (`full_insee` au-delà de 100k lignes).
2. **Retrieval Hybride & Sauvetage Déterministe** :
   - TF-IDF local sur les noms (mots 1-2, caractères 3-5) et sur les voies.
   - Whitelists universelles de sauvetage (*Universal Rescue*) par `addr_hash` MD5 et tokens numériques.
3. **Architecture 3-Stages XGBoost** :
   - **Stage 1 (Fast Ranker)** : Élagage top-50 en < 5ms (33 features non-sémantiques).
   - **Stage 2 (Semantic Decider)** : Scoring CamemBERT vectoriel sous Semantic Gate + 68 features complètes.
   - **Stage 3 (Risk Metamodel)** : Estimation isotonique de probabilité d'erreur sur la scène de décision (seuil AUTO 0.835).
4. **Fallback Places as CRM Repair** :
   - Aiguillage des cas `REVIEW` vers Google Places avec Dept-Guard strict.
   - Ré-exécution complète du pipeline XGBoost pour validation binaire (`MATCH_PLACES` ou `NO_MATCH`).

---

## 3. Artefacts & Commits Historiques Clés
* Commits majeurs : `9ab297e`, `35fc3a3`, `a309a7c`, `66b5b87`.
* Rapport de référence : `reports/latest_v6a_sota/benchmark_v6a_eval_dual.json`.
