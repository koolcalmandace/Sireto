# SIRETO Handover - 20 Aout 2026

## Etat des lieux
Le pipeline cible n'est plus "Route B full SIREN-first" comme chemin principal.
La trajectoire retenue est desormais **V8b = V7 + SIREN expansion post-prefilter**:
- prefilter V7 SIRET local (INSEE/CP) conserve comme seed robuste
- expansion SIREN ensuite (local + cross-partition) via `siren_to_geo.parquet`
- Stage 1/2/3 inchanges structurellement, mais a reentrainer/recalibrer

Route B (ranking SIREN global en phase 1) reste dans le code pour A/B tests, mais n'est plus la strategie par defaut.

**Branche active** : `feature/geo-resolution-and-crm-expansion` (depuis `main`)

## Actions terminees (fenetre recente)
- **Chantier 1 : Resolution Geo Propre** : ajout `load_with_geo_resolution()` + `_discover_insee_codes_from_cp()` dans `partitioned_store.py` ; branchement dans `_build_candidate_pool()` de `infer.py`. Hierarchie stricte : INSEE -> CP child discovery -> LOG_EMPTY. *(commit GitHub: `c754b97`)*
- **Chantier 2a : Audit qualite increment CRM 20260817** : 15 516 cas UNSEEN valides, 0 ambigu, 0% chevauchement SIREN, verdict SATISFAISANT. *(commit GitHub: `c754b97`)*
- **Chantier 2b : Script de fusion controlee** : `scripts/merge_crm_increment.py` produit `data/crm_ok_gt_merged_v1.csv` (32 570 lignes, 0 doublon). *(commit GitHub: `c754b97`)*
- **Chantier 2c : Script .bat de reentrainement** : `scripts/merge_and_retrain.bat` (6 etapes : merge + siren_to_geo + gen ranker + gen decider + train S1 + train S2 + instructions S3). *(commit GitHub: `c754b97`)*
- **Documentation** : `rules.md`, `docs/weights_and_baseline_metrics_v0.md`, `docs/architecture_v0.md`, `docs/analysis/audit_crm_increment_20260817.md`. *(commit GitHub: `c754b97`)*
- **V8 features + hard negatives + hyperparams decider**: ajout de 7 features d'interaction, extension des hard negatives colocataires/homonymes/siblings, tuning decider (`lr=0.05`, `max_depth=7`, `400 rounds`). *(commit GitHub: `35fb441`)*
- **Route B (SIREN-first) implementee**: nouvel index global SIREN, nouveau module de retrieval SIREN, branchement conditionnel dans l'inference profile/engine. *(commit GitHub: `3e090b7`)*
- **Correctifs bloquants Route B**: fix DuckDB `:memory:`, fix champ CRM nom, fix filtre closed/open, ajout CLI `--siren-index` dans le generateur de samples. *(commit GitHub: `c356923`)*
- **Branchement Route B dans le retrieval partage (training)**: `build_candidate_pool()` supporte Route B via indices SIREN, propagation sequentielle + multiprocess dans `generate_training_samples_v5fast.py`. *(commit GitHub: `1305012`)*
- **Implementation V8b SIREN expansion (V7 + local + cross-partition)**: ajout Step 5 d'expansion apres prefilter, feature flag, cap pool dedie et telemetrie d'expansion. *(commit GitHub: `9c0e806`)*
- **Correctifs critiques V8b**: exclusion explicite Route B quand expansion activee, filtres metier expansion, recalc GT coverage/loss reason post-expansion. *(commit GitHub: `f1fbbb8`)*
- **Fix expansion SIREN en mode geo-only**: chargement des index dissocie (global vs geo) dans le generateur de samples pour eviter la desactivation silencieuse de l'expansion quand seul `siren_to_geo.parquet` est present. *(commit GitHub: `c961371`)*

## Historique structurant (deja en place)
- **Retrieval hybride sparse+dense + cache TF-IDF persistant + timing**: integration du socle P0/P1. *(commit GitHub: `9ab297e`)*
- **Ablation dense-only corrigee + flag sparse explicite**: alignement des modes retrieval et signature de config. *(commit GitHub: `35fc3a3`)*
- **Defaults partitions V7 + manifest INSEE O(1)**: bascule des chemins/scripts vers `data/candidates_v7_all`. *(commit GitHub: `a309a7c`)*
- **Priorisation mega-communes embeddings**: orchestration dense amelioree pour runs longs. *(commit GitHub: `66b5b87`)*

## Fichiers modifies recemment
- `src/xgb_matcher/features.py` *(commit GitHub: `35fb441`)*
- `scripts/generate_training_samples_v5fast.py` *(commits GitHub: `35fb441`, `c356923`, `1305012`, `c961371`)*
- `scripts/train_xgb_decider.py` *(commit GitHub: `35fb441`)*
- `scripts/build_siren_global_index.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/siren_retrieval.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/infer.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/retrieval.py` *(commits GitHub: `1305012`, `9c0e806`, `f1fbbb8`)*
- `src/xgb_matcher/retrieval_config.py` *(commits GitHub: `3e090b7`, `9c0e806`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*

## Travail en cours
- **Execution locale du script merge_and_retrain.bat** : a lancer par l'utilisateur pour generer crm_ok_gt_merged_v1.csv + samples + retrain Stage 1+2.
- **Recalibration Stage 3** : apres avoir valide S1+S2 sur dev set, recalibrer le Risk Model et mettre a jour le seuil AUTO.
- **A/B de verification** : garder Route B disponible uniquement pour comparaison offline.
- **Evaluation post-fusion** : benchmark complet sur les 32 570 cas (coverage pool, Hit@1, latence, segment PRUNED/NIP).

## Points d'attention
- **Clarification nomenclature**: "V8" dans les echanges = V8b (V7 + SIREN expansion), pas Route B full.
- **Validation metrique manquante**: pas encore de benchmark consolide post-V8b.
- **Latence expansion**: mesurer sur commune dense (impact des loads cross-partition via cache INSEE).
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V8b)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Mapping geo SIREN (obligatoire V8b) | `data/siren_index/siren_to_geo.parquet` |
| Index SIREN global (optionnel A/B Route B) | `data/siren_index/word_matrix.npz` + `char_matrix.npz` |
| Samples decider V8b | `data/samples_v8b_decider.parquet` |
| Ranker V8b | `models/v8b_ranker*.json` |
| Decider V8b | `models/v8b_decider*.json` |
| Meta two-stage | `models/xgb_two_stage_meta_*.json` |

## Prochaines etapes
1. Lancer `scripts/merge_and_retrain.bat` en local (pipeline complet 6 etapes).
2. Verifier `data/crm_ok_gt_merged_v1.csv` : 32 570 lignes attendues.
3. Apres generation des samples, verifier coverage GT pool sur le dev set (cible : >98.8%).
4. Reentrainer Stage 1 puis Stage 2 sur les nouveaux samples.
5. Refaire l'evaluation complete post-fusion (Hit@1, latence, AUTO rate).
6. Recalibrer Stage 3 (`routing_risk_model.pkl`) avec le nouveau seuil AUTO/REVIEW et commiter.
7. Pousser la branche `feature/geo-resolution-and-crm-expansion` sur GitHub et ouvrir une PR vers main.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
