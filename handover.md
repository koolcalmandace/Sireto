# SIRETO Active Handover Cockpit — 7 Septembre 2026

**Branche active** : `feature/geo-resolution-and-crm-expansion` (issue de `main`)  
**Phase courante** : [Phase 3 : Résolution Géo Stricte & Expansion CRM 32k](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)  
**Cartographie documentaire complète** : [docs/INDEX.md](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/INDEX.md)

---

## 🎯 Séquence Active d'Exécution (Pas-à-Pas)

| Étape | Description & Rôle | Statut | Livrables & Artefacts |
| :---: | :--- | :---: | :--- |
| **Étape 1** | **Unification Résolution Géo** (`load_with_geo_resolution`) | ✅ **Validée** | [`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py), [`infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py) *(commit `fc4396b`)* |
| **Étape 2** | **Validation Empirique 17k Baseline** (Zéro Régression) | ✅ **Validée** | [`geo_benchmark_17k_baseline_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_17k_baseline_summary.json), [`analysis_step2_*.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step2_geo_benchmark_17k_baseline.md) |
| **Étape 3** | **Validation Empirique Incrément 15 516** | 🔄 **En cours (~25% / 5k+ requêtes)** | [`run_geo_benchmark_increment.bat`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/run_geo_benchmark_increment.bat), [`geo_benchmark_15k_increment_details.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_15k_increment_details.csv) |
| **Étape 4** | **Fusion Contrôlée des Datasets (32 570)** | ⏳ **En attente** | [`merge_crm_increment.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/merge_crm_increment.py) $\rightarrow$ `data/crm_ok_gt_merged_v1.csv` |
| **Étape 5** | **Génération Store SIRENE National V8 (France Entière)** | ✅ **Terminée (29 219 communes)** | [`build_candidate_partitions_national.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/build_candidate_partitions_national.py) $\rightarrow$ `data/candidates_v8_national/` |
| **Étape 6** | **Réentraînement XGBoost v2 (Ranker, Decider, Risk)** | ⏳ **Feuille de route cadrée** | Voir décisions clés ci-dessous |

---

## 📌 Décisions Stratégiques Cadrées pour la Suite (Réentraînement XGBoost)

1. **Extension & Rafraîchissement SIRENE National** :
   - Génération des partitions Parquet pour la totalité des communes françaises (comblant les ~94% de `NOT_IN_PARTITION` avec `base_pool == 0` observés sur les nouvelles communes de l'incrément).
   - Padding strict des codes INSEE à 5 chiffres (`.zfill(5)`).
   - Prise en compte de la date du snapshot pour intégrer les créations récentes.
2. **Optimisation Qualité TF-IDF à $K$ constant ($K=500$)** :
   - *Field-weighting* : surpondération de la raison sociale/enseigne par rapport à l'adresse.
   - Atténuation des stopwords juridiques/génériques (*SARL, SAS, SCI, ETS, SOCIETE, MAIRIE*).
   - Priorisation déterministe des établissements actifs (`etat_admin == 'A'`) et sièges à score textuel équivalent.
3. **Règle d'Or de Réentraînement ML** :
   - Exclusion stricte de l'entraînement des cas sans Ground Truth dans la partition (pour ne pas biaiser la calibration des scores).
   - Découpage Train/Dev/Test étanche par **`GroupKFold(siren)`** garantissant zéro *data leakage*.

---

## ⚡ Optimisations Haute Performance du Store & Caches *(Commits `61bf67b`, `0925ba9`, `effffbc`)*
- **Standard Officiel** : [`docs/architecture/store_and_cache_high_performance_standard.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/architecture/store_and_cache_high_performance_standard.md)
- **Accès direct aux partitions Parquet $O(1)$** ([`partitioned_store.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/partitioned_store.py)) : Remplacement du scan récursif `pyarrow.dataset` par une lecture directe ciblée du fichier `insee=XXXXX/*.parquet` (< 2ms par commune) et résolution directe CP $\rightarrow$ INSEE sans scan.
- **Suppression Définitive de la Duplication Parquet & Verrou de Store** *(Commit `effffbc`)* :
  - Élimination des 699 fichiers Parquet dupliqués dans `data/candidates_v7_all/insee/` (générés historiquement par PyArrow `write_to_dataset`).
  - Ajout d'un garde-fou applicatif de dédoublonnage par `siret` dans `load_by_insee()` et `load_by_postcode()`.
  - Nombre de candidats en mémoire divisé par 2 (de 4M à 2M de dictionnaires sur les métropoles), bornant la RAM active sous **~4.5 Go** avec les 300 communes en cache (hit rate optimal de 49.8%).
- **Cache RAM équilibré & LRU strict (300 communes)** ([`evaluate_geo_resolution.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/evaluate_geo_resolution.py)) : Capacité calibrée à 300 communes avec politique d'éviction LRU stricte (`OrderedDict.popitem(last=False)`) et `move_to_end` à l'accès.
- **Durcissement Mémoire & Résilience Veille** *(Commit `0925ba9`)* :
  - Encapsulation intégrale des compréhensions de listes dans `build_address_tfidf_index` et `build_word_tfidf_index` ([`blocking.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/blocking.py)) sous `try...except (ValueError, MemoryError, Exception)`, éliminant tout crash OOM sur Paris ou les grandes métropoles.
  - Clé canonique unifiée de partition (`f"insee_{code_insee}"`) dans `build_candidate_pool` ([`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py)), garantissant zéro duplication intra-commune dans `tfidf_cache`.
  - Élimination de la liste brute de chaînes `names` dans `_build_tfidf_artifacts` (libérant ~2.5 à 3.5 Go de RAM en ne conservant que les matrices CSR et vectoriseurs).
  - Empreinte RAM stabilisée sous **~4.5 Go** (au lieu de 9.5 Go), garantissant une résilience totale lors de la mise en veille de l'OS.
- **Plafonnement IDF Map 20k & Fail-Safe** ([`candidates.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/candidates.py)) : Échantillonnage déterministe à 20 000 candidats max et capture `(MemoryError, Exception)` évitant les crashs sur Rennes (275k lignes) ou Paris.
- **Sécurité Mémoire Char TF-IDF Bounded & Safe** ([`blocking.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/blocking.py)) : Plafonnement `max_features=50_000`, `min_df` adaptatif et capture `MemoryError` évitant tout `std::bad_alloc` sur les grandes métropoles.
- **Exécution 100% Mono-cœur Sécurisée** : Zéro crash multiprocessing, empreinte CPU minimale et stabilité absolue.

---

## 🛡️ Architecture de Sécurité & Résilience des Benchmarks *(Commit `f1b4854`)*
- **Standard Officiel** : [`docs/architecture/safe_batch_execution_standard.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/architecture/safe_batch_execution_standard.md)
- **Checkpoint Atomique JSON (`CheckpointManager`)** : Persistance du progrès par requête (`.checkpoints/checkpoint_<tag>.json`), reprise automatique (`--resume`) en cas d'interruption, et streaming direct des lignes dans les CSV de détails.
- **Moniteur PowerGuard (`PowerMonitor`)** : 
  - Pause propre sur verrouillage/inactivité (0% CPU) et reprise automatique.
  - Résilience totale validée lors des interruptions système.

---

## 📜 Historisation Séquentielle des Phases Passées
* 📂 [Phase 1 : Transition Pipe V6 (LLM) vers Pipe V7 Déterministe](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase1_pipe_v6_to_v7_deterministic.md)
* 📂 [Phase 2 : Exploration Route B (SIREN-First) & Pivot V8b](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase2_v8_route_b_and_pivot_v8b.md)
* 📂 [Phase 3 : Résolution Géo Stricte & Expansion CRM (32k)](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)
* 📂 [Registre Général des Versions & Jalons Git](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/branch_history_and_milestones.md)
