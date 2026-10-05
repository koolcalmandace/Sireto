# SIRETO Active Handover Cockpit — 28 Septembre 2026

**Branche active** : `feature/holding-establishment-reconciliation` (issue de `feature/semantic-vector-sidecar-and-naf-projection`)  
**Phase courante** : [Phase 3b : Prototypes Sémantiques TF-IDF, Sidecar NAF & Intégrations Rescapées](docs/history/phase3_geo_resolution_and_crm_expansion.md)  
**Cartographie documentaire complète** : [docs/INDEX.md](docs/INDEX.md)

---

## 🎯 Séquence Active d'Exécution (Pas-à-Pas)

| Étape | Description & Rôle | Statut | Livrables & Artefacts |
| :--- | :--- | :---: | :--- |
| **Étape 1** | **Unification Résolution Géo** (`load_with_geo_resolution`) | ✅ **Validée** | [`retrieval.py`](src/xgb_matcher/retrieval.py), [`infer.py`](src/xgb_matcher/infer.py) *(commit `fc4396b`)* |
| **Étape 2** | **Validation Empirique 17k Baseline** (Zéro Régression) | ✅ **Validée** | [`geo_benchmark_17k_baseline_summary.json`](reports/geo_validation/geo_benchmark_17k_baseline_summary.json), [`analysis_step2_*.md`](docs/analysis/analysis_step2_geo_benchmark_17k_baseline.md) |
| **Étape 3** | **Validation Empirique Incrément 15 516** | ✅ **Validée (100% / 20 209 requêtes)** | [`geo_benchmark_15k_increment_summary.json`](reports/geo_validation/geo_benchmark_15k_increment_summary.json), [`analysis_step3_*.md`](docs/analysis/analysis_step3_geo_benchmark_15k_increment.md), [`analysis_step3_pruned_cases_breakdown.md`](docs/analysis/analysis_step3_pruned_cases_breakdown.md) *(commit `61dfa1e`)* |
| **Étape 4** | **Fusion Contrôlée des Datasets (32 570)** | ✅ **Validée (32 570 requêtes)** | [`merge_crm_increment.py`](scripts/merge_crm_increment.py), [`data/crm_ok_gt_merged_v1.csv`](data/crm_ok_gt_merged_v1.csv) *(commit `61dfa1e`)* |
| **Étape 5** | **Store National V8 & Nouvelles Règles de Retrieval** | ✅ **Validée (70.14% / +3 859 requêtes)** | [`geo_benchmark_v8_recovery_5502_summary.json`](reports/geo_validation/geo_benchmark_v8_recovery_5502_summary.json), [`analysis_step5_v8_recovery_5502.md`](docs/analysis/analysis_step5_v8_recovery_5502.md) *(commit `c77793b`)* |
| **Étape 5b**| **Prototypes Sémantiques Étendus & Réparation Partitions** | ✅ **Validée (88.59% Top-500 / 97.55% Base Pool)** | [`geo_benchmark_v8_recovery_enhanced_summary.json`](reports/geo_validation/geo_benchmark_v8_recovery_enhanced_summary.json), [`analysis_step5b_v8_recovery_enhanced.md`](docs/analysis/analysis_step5b_v8_recovery_enhanced.md), [`repair_missing_v8_departments.py`](scripts/repair_missing_v8_departments.py), [`retrieval.py`](src/xgb_matcher/retrieval.py) *(commits `f386cd0`, `c99126b`)* |
| **Étape 5c**| **Certification Finale Retrieval 32k (V8 National)** | ✅ **Validée (95.92% @500 / 98.26% Base / 32 884 reqs)** | Évaluation intégrale sur 32 884 requêtes via [`run_geo_benchmark_32k.bat`](scripts/run_geo_benchmark_32k.bat) (Global: 95.92% @500, 98.26% Base. Actifs: 98.10% @500, 99.90% Base. Fermés: 86.63% @500, 91.29% Base) *(commits `2ed7874`, `fe88483`, `85cddba`)* |
| **Étape 5d**| **Stopwords Juridiques Métropole & Priorité Enseigne TF-IDF** | ✅ **Implémentée & Testée (100%)** | Élimination de l'éviction en métropole (`LEGAL_CORPORATE_STOPWORDS`), surpondération d'enseigne dans `candidate_tfidf_text`, expansion sécurité sociale (84/85) et médico-sociale (9210) *(commits `95c45b3`, `49be97e`)* |
| **Étape 5e**| **Whitelisting Établissements Secondaires Communaux & Extension Équipements Publics** | ✅ **Validée (100% Tests Passés)** | Dé-concaténation alphanumérique des requêtes CRM, whitelisting inconditionnel des établissements communaux (`cj_ul == 7210`) sur requêtes publiques (résolution écoles/gymnases/annexes), couverture étendue médico-sociale, scolaire et territoriale *(commit `a7cc1e4`)* |
| **Étape 5f**| **Module Satellite Découplé (Sidecar d'Entités 3 Piliers)** | ✅ **Créé & Testé en Isolation (100%)** | Module autonome [`sidecar.py`](src/xgb_matcher/sidecar.py) (Initiales dynamiques, NAF INSEE, Enseignes sans Google) et suite de tests [`test_entity_sidecar.py`](tests/test_entity_sidecar.py) *(commit `6ee36a1`)* |
| **Étape 5g**| **Dé-concaténation Alphanumérique Universelle & Acronymes Résilients aux Majuscules** | ✅ **Validée (100% Tests Passés)** | Dé-concaténation générique `([a-zA-Z]+)(\d+)` étendue à l'ensemble du TF-IDF (`blocking.py`) et du Sidecar (`sidecar.py`). Extraction d'acronymes résiliente aux exports CRM 100% majuscules *(commit `42922db`)* |
| **Étape 5h**| **Intégrations Rescapées North Star & Injection Whitelist Sidecar** | ✅ **Validée (100% Tests Passés / 38 Tests)** | Correction bug historique acronymes C5, normalisation types de voie BAN, multi-représentation racines délimitées, intervalles et positions multiples de numéros de voie, feature copropriété (`cj_ul == 9110`), moteur vectoriel NAF Pilier 3, et injection de sauvetage Sidecar dans le retrieval ([`features.py`](src/xgb_matcher/features.py), [`retrieval.py`](src/xgb_matcher/retrieval.py), [`naf_vector_engine.py`](src/xgb_matcher/naf_vector_engine.py), [`rescued_integrations_spec.md`](docs/architecture/rescued_integrations_spec.md)) *(commit `4e4d9fc`)* |
| **Étape 5i**| **Moteur Universel SSDA (Double-Ancre Spatiale-Sectorielle)** | ✅ **Validée (100% Tests Passés / 42 Tests)** | Éradication complète des dictionnaires statiques d'enseignes (`NATIONAL_FRANCHISE_NETWORKS`), mise en place du moteur généraliste Double-Ancre Spatiale (adresse BAN stricte) et Sectorielle (projection vectorielle continue NAF INSEE), réconciliation métaclasse holding (`64.20`, `70.10`, `68.20`, `82.99`), garde d'exclusivité mono-occupant (`density == 1`) et suite de tests durcie ([`sidecar.py`](src/xgb_matcher/sidecar.py), [`test_entity_sidecar.py`](tests/test_entity_sidecar.py)) *(commit `73b9da0`)* |
| **Étape 5j**| **Réconciliation Établissement vs Holding & Schéma 4 Colonnes** | ✅ **Validée (158/459 cas / +34.4%)** | Branche `feature/holding-establishment-reconciliation` : qualification native 4 colonnes (`siret_retenu`, `type_etablissement`, `siret_siege_holding`, `nom_holding`), prise en compte universelle des personnes physiques (`nom_ul`, `prenom_usuel_ul`), pondération de spécificité lexicale et marge anti-collision multi-occupants, benchmark 459 cas prunés portant les qualifications à 158 cas (hausse rappel actif 32k de 98.10% à 98.69%), 49/49 tests unitaires au vert ([`sidecar.py`](src/xgb_matcher/sidecar.py), [`infer.py`](src/xgb_matcher/infer.py), [`test_entity_sidecar.py`](tests/test_entity_sidecar.py), [`analysis_holding_reconciliation_benchmark_459.md`](docs/analysis/analysis_holding_reconciliation_benchmark_459.md)) *(commits `2c41678`, `effb4ec`)* |
| **Étape 5k**| **Communes Nouvelles & Ancrage Toponymique sans Heuristique Arbitraire** | ✅ **Validée (163/459 cas / +35.51% - Rappel Actif 32k @ 99.10%)** | Intégration stricte et découplée dans le Sidecar : table de passage officielle INSEE COG (`communes_nouvelles.py`), ancrage toponymique déterministe (sans seuil de population ni limitation NAF arbitraire) fondé sur l'exclusivité $N=1$ et la marge lexicale $\Delta \ge 1.5$, benchmark 459 cas portant à 163 qualifiés, 25/25 tests Sidecar et 97/97 tests globaux au vert ([`communes_nouvelles.py`](src/xgb_matcher/communes_nouvelles.py), [`sidecar.py`](src/xgb_matcher/sidecar.py), [`test_entity_sidecar.py`](tests/test_entity_sidecar.py)) *(commit `eceed01`)* |
| **Étape 5l**| **Validation Empirique 32k avec Sidecar Complet (Stage 1 Certifié)** | ✅ **Validée (98.16% Actif @500 / 99.90% Base / 33 011 reqs)** | Évaluation intégrale sur 33 011 requêtes via [`run_geo_benchmark_32k_sidecar.bat`](scripts/run_geo_benchmark_32k_sidecar.bat) (Global: 95.95% @500, 98.26% Base. Actifs: 98.16% @500, 99.90% Base. 24 cas emblématiques repêchés dont personnes physiques et enseignes nationales, zéro crash/fuite mémoire sous PowerGuard) *(commits `3687286`, `3c7ac5e`, rapports [`analysis_step5l_*.md`](docs/analysis/analysis_step5l_geo_benchmark_32k_sidecar.md), [`analysis_sirene_ceiling_and_pruned_cases_762.md`](docs/analysis/analysis_sirene_ceiling_and_pruned_cases_762.md) commit `863b7ca`)* |
| **Étape 6a**| **Vectorisation Matricielle des Règles (Axe 2 - Blocs Tensoriels & Zéro Dérive)** | ✅ **Validée (10 000 cands / 200 reqs)** | Regroupement des 55 règles en 4 blocs vectoriels continus ([`make_features_batch`](src/xgb_matcher/features.py)), parité numérique stricte ($\max \Delta = 0.0000000000$), standard documenté ([`matrix_vector_rules_standard.md`](docs/architecture/matrix_vector_rules_standard.md)) *(commit `c9deee4`)* |
| **Étape 6b**| **Rétention & Calibration Stage 1 (Fast Ranker 38 features & Benchmark 32k)** | ⏳ **Modèle Entraîné / Prêt pour Benchmark** | Branche `experiment/fast-ranker-32k-retention` : décontamination Sidecar, ajout `is_siege` & `addr_unsupported_by_name`, rétention Top-50 (98.81% rétention active sur test), modèle `models/xgbranker_fast_v8.json` finalisé (1917 train / 965 dev, NDCG@50=0.822) et prêt pour Safe Batch 32k ([`fast_ranker_32k_modifications_spec.md`](docs/architecture/fast_ranker_32k_modifications_spec.md), [`evaluate_fast_ranker_32k.py`](scripts/evaluate_fast_ranker_32k.py)) *(commits `cbda765`, `f178228`)* |
| **Étape 6c**| **Réentraînement & Calibration Stage 2 (Decider V8 National)** | ⏳ **À venir** | Probabilité binaire de match, équilibrage pondéré `scale_pos_weight` |
| **Étape 6d**| **Réentraînement & Calibration Stage 3 (Risk Model & Détecteur d'Incertitude)** | ⏳ **À venir** | Régression isotonique de scène, recalibration seuil AUTO (`0.835`) et détection d'absence GT |
| **Étape 6e**| **Validation Finale Double Échantillon Aléatoire Indépendant (Certification)** | ⏳ **À venir** | Exécution intégrale bout en bout 2 fois sur des samples aléatoires indépendants |




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
   - Maintien du seuil de 10 ans : les établissements fermés avant 2016 sont exclus du store pour éviter d'encombrer l'espace de recherche avec des entités éteintes (99.4% des fermés non-retrouvés relèvent de ce filtre légitime).
   - *Règle Métier CRM* : Le CRM reflétant un client/service actif bien connecté, une radiation SIRENE signale généralement une reprise/transmission de fonds, un changement de raison sociale sous le même nom/adresse, ou un déménagement. Le système privilégie la détection de l'établissement actif successeur (via l'adresse et le fallback Places) plutôt que le rattachement à un SIRET juridiquement éteint.
6. **Architecture Généraliste du Sidecar (Double-Ancre Spatiale-Sectorielle — SSDA)** *(Cadrée le 22/09)* :
   - **Abandon de la stratégie de dictionnaires statiques** (`NATIONAL_FRANCHISE_NETWORKS`), jugée fragile et sujette à surspécialisation.
   - **Méthode Universelle 100% Déterministe** : Inversion double-ancre reposant exclusivement sur les données natives CRM/SIRENE :
     - *Ancre Spatiale* : Numéro de voie exact + Jaro voie $\ge 0.85$ (BAN-normalisé).
     - *Ancre Sectorielle* : Projection vectorielle continue NAF INSEE (via [`naf_vector_engine.py`](src/xgb_matcher/naf_vector_engine.py)).
     - *Garde d'Exclusivité* : Whitelisting si et seulement si $N_{\text{secteur, adresse}} == 1$ (évite toute collision en centre commercial).
     - *Réconciliation NAF Holding / Exploitation* : Prise en charge des divergences via Dual-NAF (Établissement vs UL), métaclasse transversale holding (`64.20`, `70.10`, `68.20`), empreinte globale SIREN et liens de gouvernance (`pm_dirigeant_names`).
     - *Traitement des [ND] (RGPD)* : Résolution native des professionnels libéraux/médicaux masqués via l'exclusivité spatiale + NAF sans nécessiter de chaîne textuelle.
7. **Schéma d'Output Unifié Établissement vs Holding (Clarté Opérationnelle & Audit)** :
   - Élimination de toute confusion pour l'utilisateur final et lors des recherches indépendantes.
   - Les 4 colonnes canoniques de qualification du livrable XGBoost :
     - `siret_retenu` : SIRET attribué au bâtiment physique (sur site).
     - `type_etablissement` : Qualification de l'entité (`SIEGE`, `SECONDAIRE`, `FILIALE`).
     - `siret_siege_holding` : SIRET du siège social ou de la holding parente.
     - `nom_holding` : Raison sociale légale de la société mère / holding.
8. **Assainissement ML Pré-Étape 6 & Bannissement des Heuristiques Post-Hoc** *(Commit `f935a85`)* :
   - Éradication formelle de la règle post-hoc `_promote_open_over_closed` dans `infer.py` et `train_xgb_decider.py` : le classement est désormais 100% data-driven par le gradient boosting XGBoost sans aucune permutation manuelle.
   - Élimination des listes statiques de tokens (`ASSOCIATION_TOKENS`, `SCHOOL_TOKENS` dans `features.py`) au profit de la classification officielle INSEE `cj_ul.startswith("92")` et de la projection continue vectorielle NAF.
   - Validation mathématique de `is_copropriete` (`cj_ul == '9110'`) comme feature continue d'arbre complétée par la macro-catégorie `legal_form_category` sans exclusion dure.

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
