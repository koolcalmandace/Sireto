# SIRETO Active Handover Cockpit — 4 Septembre 2026

**Branche active** : `feature/geo-resolution-and-crm-expansion` (issue de `main`)  
**Phase courante** : [Phase 3 : Résolution Géo Stricte & Expansion CRM 32k](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)  
**Cartographie documentaire complète** : [docs/INDEX.md](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/INDEX.md)

---

## 🎯 Séquence Active d'Exécution (Pas-à-Pas)

| Étape | Description & Rôle | Statut | Livrables & Artefacts |
| :---: | :--- | :---: | :--- |
| **Étape 1** | **Unification Résolution Géo** (`load_with_geo_resolution`) | ✅ **Validée** | [`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py), [`infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py) *(commit `fc4396b`)* |
| **Étape 2** | **Validation Empirique 17k Baseline** (Zéro Régression) | ✅ **Validée** | [`geo_benchmark_17k_baseline_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_17k_baseline_summary.json), [`analysis_step2_*.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/analysis_step2_geo_benchmark_17k_baseline.md) |
| **Étape 3** | **Validation Empirique Incrément 15 516** | 🔄 **En cours** | [`run_geo_benchmark_increment.bat`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/run_geo_benchmark_increment.bat) *(commit `f1b4854`)* |
| **Étape 4** | **Fusion Contrôlée des Datasets (32 570)** | ⏳ **En attente** | [`merge_crm_increment.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/merge_crm_increment.py) $\rightarrow$ `data/crm_ok_gt_merged_v1.csv` |
| **Étape 5** | **Audit Approfondi de l'Architecture & Acronymes** (Étapes 2 à 7) | ⏳ **En attente** | Intégration acronymes/sigles, analyses dans `docs/analysis/` |
| **Étape 6** | **Réentraînement des Poids & Recalibration Risk** | ⏸️ **Standby** | [`merge_and_retrain.bat`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/merge_and_retrain.bat) (arbitrage multi-branches) |

---

## 🛡️ Architecture de Sécurité & Résilience des Benchmarks *(Commit `f1b4854`)*
- **Standard Officiel** : [`docs/architecture/safe_batch_execution_standard.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/architecture/safe_batch_execution_standard.md)
- **Checkpoint Atomique JSON (`CheckpointManager`)** : Persistance du progrès par requête (`.checkpoints/checkpoint_<tag>.json`), reprise automatique (`--resume`) en cas d'interruption, et streaming direct des lignes dans les CSV de détails.
- **Moniteur PowerGuard (`PowerMonitor`)** : 
  - **Pause sur verrouillage manuel (`pause_on_lock=True`)** : Détection active du verrouillage (`Win+L`), rabat du capot avec flush instantané du checkpoint et passage à 0% CPU.
  - **Détection d'inactivité utilisateur 5 minutes (`idle_timeout_seconds=300.0`)** : Détection de l'inactivité clavier/souris (`GetLastInputInfo`) pour céder le CPU à 0% avant la mise en veille Modern Standby Windows (10 min), évitant tout conflit de tension électrique ou plantage matériel.
  - **Inhibition veille intempestive (`SetThreadExecutionState`)** : Empêche l'endormissement intempestif de Windows pendant le calcul actif (écran ouvert et activité), et libère l'état pour une veille sécurisée dès que la session est inactive ou verrouillée.
  - **Reprise automatique** : Redémarrage immédiat des calculs dès le déverrouillage de la session ou la détection d'une frappe clavier/souris sans intervention manuelle.

---

## 📜 Historisation Séquentielle des Phases Passées
Pour consulter l'historique détaillé des décisions majeures, jalons et architecture :
* 📂 [Phase 1 : Transition Pipe V6 (LLM) vers Pipe V7 Déterministe](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase1_pipe_v6_to_v7_deterministic.md)
* 📂 [Phase 2 : Exploration Route B (SIREN-First) & Pivot V8b](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase2_v8_route_b_and_pivot_v8b.md)
* 📂 [Phase 3 : Résolution Géo Stricte & Expansion CRM (32k)](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/phase3_geo_resolution_and_crm_expansion.md)
* 📂 [Registre Général des Versions & Jalons Git](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/history/branch_history_and_milestones.md)

---

## 🚀 Prochaine Action Immédiate
Exécution en cours du benchmark incrément via [`scripts/run_geo_benchmark_increment.bat`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/run_geo_benchmark_increment.bat) pour générer `reports/geo_validation/geo_benchmark_15k_increment_summary.json` et valider la généralisation sur les 15 516 requêtes unseen.
