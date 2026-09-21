# SIRETO Active Handover Cockpit — 21 Septembre 2026

**Branche active** : `feature/retrieval-public-communal-whitelisting` (issue de `feature/tfidf-metropole-stopwords-and-enseigne-priority`)  
**Phase courante** : [Phase 3b : Prototypes Sémantiques TF-IDF & Complétion Partitions V8](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)  
**Cartographie documentaire complète** : [docs/INDEX.md](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/INDEX.md)

---

## 🎯 Séquence Active d'Exécution (Pas-à-Pas)

| Étape | Description & Rôle | Statut | Livrables & Artefacts |
| :--- | :--- | :---: | :--- |
| **Étape 1** | **Unification Résolution Géo** (`load_with_geo_resolution`) | ✅ **Validée** | [`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py), [`infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py) *(commit `fc4396b`)* |
| **Étape 2** | **Validation Empirique 17k Baseline** (Zéro Régression) | ✅ **Validée** | [`geo_benchmark_17k_baseline_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_17k_baseline_summary.json), [`analysis_step2_*.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step2_geo_benchmark_17k_baseline.md) |
| **Étape 3** | **Validation Empirique Incrément 15 516** | ✅ **Validée (100% / 20 209 requêtes)** | [`geo_benchmark_15k_increment_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_15k_increment_summary.json), [`analysis_step3_*.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step3_geo_benchmark_15k_increment.md), [`analysis_step3_pruned_cases_breakdown.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step3_pruned_cases_breakdown.md) *(commit `61dfa1e`)* |
| **Étape 4** | **Fusion Contrôlée des Datasets (32 570)** | ✅ **Validée (32 570 requêtes)** | [`merge_crm_increment.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/merge_crm_increment.py), [`data/crm_ok_gt_merged_v1.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/data/crm_ok_gt_merged_v1.csv) *(commit `61dfa1e`)* |
| **Étape 5** | **Store National V8 & Nouvelles Règles de Retrieval** | ✅ **Validée (70.14% / +3 859 requêtes)** | [`geo_benchmark_v8_recovery_5502_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_v8_recovery_5502_summary.json), [`analysis_step5_v8_recovery_5502.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step5_v8_recovery_5502.md) *(commit `c77793b`)* |
| **Étape 5b**| **Prototypes Sémantiques Étendus & Réparation Partitions** | ✅ **Validée (88.59% Top-500 / 97.55% Base Pool)** | [`geo_benchmark_v8_recovery_enhanced_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_v8_recovery_enhanced_summary.json), [`analysis_step5b_v8_recovery_enhanced.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step5b_v8_recovery_enhanced.md), [`repair_missing_v8_departments.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/repair_missing_v8_departments.py), [`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py) *(commits `f386cd0`, `c99126b`)* |
| **Étape 5c**| **Certification Finale Retrieval 32k (V8 National)** | 🔄 **En cours d'exécution** | Évaluation intégrale sur les 32 570 requêtes fusionnées via [`run_geo_benchmark_32k.bat`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/run_geo_benchmark_32k.bat) (Dpts 24/66 restaurés, whitelisting communal actif, PowerGuard + Checkpointing) *(commits `2ed7874`, `fe88483`, `85cddba`)* |
| **Étape 5d**| **Stopwords Juridiques Métropole & Priorité Enseigne TF-IDF** | ✅ **Implémentée & Testée (100%)** | Élimination de l'éviction en métropole (`LEGAL_CORPORATE_STOPWORDS`), surpondération d'enseigne dans `candidate_tfidf_text`, expansion sécurité sociale (84/85) et médico-sociale (9210) *(commits `95c45b3`, `49be97e`)* |
| **Étape 5e**| **Whitelisting Établissements Secondaires Communaux & Extension Équipements Publics** | ✅ **Validée (100% Tests Passés)** | Dé-concaténation alphanumérique des requêtes CRM, whitelisting inconditionnel des établissements communaux (`cj_ul == 7210`) sur requêtes publiques (résolution écoles/gymnases/annexes), couverture étendue médico-sociale, scolaire et territoriale *(commit `a7cc1e4`)* |
| **Étape 5f**| **Module Satellite Découplé (Sidecar d'Entités 3 Piliers)** | ✅ **Créé & Testé en Isolation (100%)** | Module autonome [`sidecar.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/sidecar.py) (Initiales dynamiques, NAF INSEE, Enseignes sans Google) et suite de tests [`test_entity_sidecar.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/tests/test_entity_sidecar.py) ; maintenu strictement isolé du pipeline actif *(commit `6ee36a1`)* |
| **Étape 5g**| **Dé-concaténation Alphanumérique Universelle & Acronymes Résilients aux Majuscules** | ✅ **Validée (100% Tests Passés)** | Dé-concaténation générique `([a-zA-Z]+)(\d+)` étendue à l'ensemble du TF-IDF (`blocking.py`) et du Sidecar (`sidecar.py`). Extraction d'acronymes résiliente aux exports CRM 100% majuscules (filtrage `COMMON_FRENCH_WORDS`, délimiteurs, proximité de chiffres et matching direct `sigle_ul`) *(commit `42922db`)* |
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
4. **Gestion des Anciennes Communes & Communes Nouvelles** :
   - Intégration de la table de passage officielle INSEE (communes déléguées $\rightarrow$ communes nouvelles) comme filet de sécurité pour les nouveaux flux CRM présentant d'anciens codes.
5. **Gestion Métier des Radiations & Successions d'Entreprises** :
   - Maintien du seuil de 10 ans : les établissements fermés avant 2016 sont exclus du store pour éviter d'encombrer l'espace de recherche avec des entités éteintes.
   - *Règle Métier CRM* : Le CRM reflétant un client/service actif bien connecté, une radiation SIRENE signale généralement une reprise/transmission de fonds, un changement de raison sociale sous le même nom/adresse, ou un déménagement. Le système privilégie la détection de l'établissement actif successeur (via l'adresse et le fallback Places) plutôt que le rattachement à un SIRET juridiquement éteint.

---

## ⚡ Optimisations Haute Performance du Store & Caches *(Commits `61bf67b`, `0925ba9`, `effffbc`, `41ada61`)*
- **Standard Officiel** : [`docs/architecture/store_and_cache_high_performance_standard.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/architecture/store_and_cache_high_performance_standard.md)
- **Accès direct aux partitions Parquet $O(1)$** ([`partitioned_store.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/partitioned_store.py)) : Remplacement du scan récursif `pyarrow.dataset` par une lecture directe ciblée du fichier `insee=XXXXX/*.parquet` (< 2ms par commune) et résolution directe CP $\rightarrow$ INSEE sans scan.
- **Suppression Définitive de la Duplication Parquet & Verrou de Store** *(Commit `effffbc`)* :
  - Élimination des 699 fichiers Parquet dupliqués dans `data/candidates_v7_all/insee/` (générés historiquement par PyArrow `write_to_dataset`).
  - Ajout d'un garde-fou applicatif de dédoublonnage par `siret` dans `load_by_insee()` et `load_by_postcode()`.
  - Nombre de candidats en mémoire divisé par 2 (de 4M à 2M de dictionnaires sur les métropoles).
- **Calibration Équilibrée du Cache LRU (150 communes)** *(Commit `41ada61`)* :
  - Capacité du store et du TF-IDF calibrée à 150 communes (`--cache-size 150`), divisant par 2 la charge mémoire active sous **~4.5 à 5.0 Go** (laissant ~11 Go de RAM libre).
  - Préservation d'un hit rate solide de 34.9% avec un surcoût temporel global négligeable de **1 min 40 s** sur 14 000 requêtes (+0.3%).
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
  - **Différenciation stricte Veille vs Extinction** : Traitement distinct des événements terminaux (`CTRL_SHUTDOWN_EVENT`, `CTRL_LOGOFF_EVENT`, `CTRL_CLOSE_EVENT`) via `should_exit()` / `trigger_exit()` pour sauvegarder le checkpoint et invoquer `sys.exit(0)` immédiatement dans le délai de grâce OS (5s), éliminant tout blocage dans `wait_for_resume()` lors d'un arrêt ou redémarrage de la machine.
  - Règle d'opération : désactivation de PowerToys Awake pour préserver la détection d'inactivité Win32 (`GetLastInputInfo`).
- **Clustering Géographique In-Memory & Accélération Débit (x20)** :
  - Groupement automatique en mémoire des requêtes CRM par commune (`cluster_rows_by_locality`) sans jamais modifier les fichiers CSV sources sur disque.
  - Résorption intégrale du thrashing de cache : les requêtes d'une même commune s'exécutent consécutivement avec un hit-rate TF-IDF/Store de 100% en RAM (< 5ms par requête).
  - Débit multiplié par plus de 20 (passant de ~140 req/h à **~3 000+ req/h**).
  - Empreinte RAM ultra-basse (**~0.6 à 1.2 Go**, bien en dessous du plafond de 4.0 Go), garantissant une parfaite cohabitation avec Power BI et le système d'exploitation sur les machines 16 Go.
  - Reprise ID-centrique robuste (`sync_from_details_csv`) réconciliant automatiquement l'état sans duplication.

---

## 📜 Historisation Séquentielle des Phases Passées
* 📂 [Phase 1 : Transition Pipe V6 (LLM) vers Pipe V7 Déterministe](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase1_pipe_v6_to_v7_deterministic.md)
* 📂 [Phase 2 : Exploration Route B (SIREN-First) & Pivot V8b](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase2_v8_route_b_and_pivot_v8b.md)
* 📂 [Phase 3 : Résolution Géo Stricte & Expansion CRM (32k)](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)
* 📂 [Registre Général des Versions & Jalons Git](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/branch_history_and_milestones.md)
