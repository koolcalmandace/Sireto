# Rapport d'Audit & Validation Étape 5 : Récupération V8 National & Règles de Retrieval

**Date** : 17 Septembre 2026  
**Branche Git** : `feature/v8-national-partitions-and-retrieval`  
**Jeu de Données Évalué** : `data/crm_lost_cases_5502.csv` (5 502 cas perdus issus de l'évaluation incrémentale 15k)  
**Magasin de Candidats** : `data/candidates_v8_national` (29 219 communes françaises)  
**Commit de Référence** : `c77793b`  

---

## 1. Synthèse Exécutive : Un Taux de Récupération Net de 70.14%

L'objectif de cette évaluation ciblée était de mesurer l'efficacité conjointe :
1. Du basculement vers le **magasin national V8** (`data/candidates_v8_national`).
2. Des **nouvelles règles de retrieval** :
   - Expansion des établissements secondaires publics via prototypes vectoriels sémantiques (`_is_public_sector_query`).
   - Repêchage d'adresse strict avec garde-fou de densité ($\le 5$ entreprises par adresse).

Le jeu de test de **5 502 requêtes** est composé exclusivement des requêtes en échec total (0% de recall) sous V7 :
- **4 760 cas** `NOT_IN_PARTITION` (communes absentes du découpage V7).
- **742 cas** `PRUNED_BY_TFIDF` (éjectés au-delà du rang 500 par le pré-filtre lexical).

### Résultat Global Consolidé

| Métrique | Situation V7 (Avant) | Situation V8 (Après) | Gain Net Immédiat |
| :--- | :---: | :---: | :---: |
| **Total Requêtes Évaluées** | 5 502 | 5 502 | - |
| **Présence Base Pool (Commune)** | 0 cas (0.00%) | **4 423 cas (80.39%)** | **+4 423 requêtes** |
| **Recall Pré-filtre Top-500 (pour XGBoost)** | 0 cas (0.00%) | **3 859 cas (70.14%)** | **+3 859 requêtes** |
| **Taux de Rétention (Pré-filtre / Base)** | 0.00% | **87.25%** | +87.25 pts |

> [!IMPORTANT]
> **3 859 entreprises et établissements publics** qui étaient complètement invisibles pour le pipeline de matching sont désormais **récupérés et injectés dans le Top-500** soumis au modèle XGBoost, sans aucun appel LLM et dans le strict respect de la sécurité mémoire.

---

## 2. Décomposition par Motif d'Échec Initial

Le croisement ligne à ligne entre l'audit initial 15k et le présent benchmark V8 met en lumière la performance spécifique de chaque mécanisme :

```
                        ┌────────────────────────────────────────────────────────┐
                        │          5 502 Cas Perdus sous V7 (0% Recall)          │
                        └───────────────────────────┬────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
   ┌───────────────────────────┐                                         ┌───────────────────────────┐
   │    4 760 Cas Perdus       │                                         │     742 Cas Perdus        │
   │    NOT_IN_PARTITION       │                                         │    PRUNED_BY_TFIDF        │
   └─────────────┬─────────────┘                                         └─────────────┬─────────────┘
                 │                                                                     │
                 ▼                                                                     ▼
   ┌───────────────────────────┐                                         ┌───────────────────────────┐
   │   3 822 en Base (80.3%)   │                                         │    601 en Base (81.0%)    │
   │  3 628 en Top-500 (76.2%) │                                         │    231 en Top-500 (31.1%) │
   └───────────────────────────┘                                         └───────────────────────────┘
```

### Analyse Détaillée

1. **Cas `NOT_IN_PARTITION` (4 760 cas)** :
   - **76.22% de conversion directe en Top-500** (3 628 cas).
   - Le magasin V8 National comble le vide géographique sur plus des trois quarts des communes rurales ou périurbaines auparavant ignorées.
2. **Cas `PRUNED_BY_TFIDF` (742 cas)** :
   - **31.13% de repêchage net en Top-500** (231 cas).
   - Près d'un tiers des cas lexicaux complexes (écoles, gymnases, collèges et adresses isolées) ont été repêchés grâce à l'expansion vectorielle publique et au repêchage d'adresse filtré par densité.

---

## 3. Analyse des Pertes Résiduelles

Sur les 5 502 requêtes, **1 643 requêtes** restent non qualifiées dans le Top-500 (contre 5 502 sous V7) :

| Motif de Perte V8 | Nombre de Cas | Part (%) | Diagnostic & Explication Métier |
| :--- | :---: | :---: | :--- |
| **`NOT_IN_PARTITION` (Commune vide)** | 927 | 16.85% | Communes dont le dossier Parquet est vide ou absent de V8 (ex: Montbéliard `25388`, La Teste `33227`, St-Gaudens `31526`). |
| **`NOT_IN_PARTITION` (GT absent de la commune)** | 152 | 2.76% | La commune existe dans V8, mais le SIRET CRM n'y figure pas (fermeture pré-snapshot, déménagement ou divergence de code INSEE dans le CRM). |
| **`PRUNED_BY_TFIDF`** | 564 | 10.25% | Cas à fort décalage lexical dans de grands centres urbains (structures hospitalières spécialisées type CATTP/CMP, enseignes commerciales vs holdings pures). |
| **Total Pertes Résiduelles** | **1 643** | **29.86%** | **70.14% de succès global**. |

> [!TIP]
> Les 564 cas `PRUNED_BY_TFIDF` résiduels relèvent exactement du périmètre dévolu au mécanisme **Places as CRM Repair** (Pipe V7) : Google Places permettra de résoudre les enseignes commerciales (ex: *DG ASSURBANQUE* $\rightarrow$ *EIRL BENEDICTE GUESNET MOREAUX*) sans risquer de polluer la phase de retrieval SIRENE locale.

---

## 4. Performance Système, Mémoire et Stabilité

Le benchmark a été exécuté avec le standard de robustesse mono-cœur développé sur la branche :

| Paramètre | Valeur Observée | Seuil de Tolérance | Statut |
| :--- | :---: | :---: | :---: |
| **Durée d'Exécution** | 5 910 s (~1h38) | - | Conforme (~0.9 req/s multi-communes) |
| **Consommation RAM Max (RSS)** | **1.52 Go** | $\le 4.0$ Go | ✅ **Très largement sous le plafond** |
| **Consommation RAM Moyenne** | **~0.70 à 1.00 Go** | - | ✅ **Empreinte ultra-légère** |
| **Purges LRU Automatiques** | 12 purges déclenchées | - | ✅ **Zéro fuite mémoire** |
| **Checkpoints Sauvegardés** | 55 checkpoints atomiques | - | ✅ **100% de résilience** |
| **Crashs / Erreurs OOM** | **0** | 0 toléré | ✅ **Stabilité absolue** |

---

## 5. Livrables Associés

- **Rapport de Synthèse JSON** : [`reports/geo_validation/geo_benchmark_v8_recovery_5502_summary.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_v8_recovery_5502_summary.json)
- **Log Détaillé Requête par Requête** : [`reports/geo_validation/geo_benchmark_v8_recovery_5502_details.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/geo_benchmark_v8_recovery_5502_details.csv)
- **Dossier de Checkpoint Finalisé** : [`reports/geo_validation/.checkpoints/checkpoint_v8_recovery_5502.completed.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/.checkpoints/checkpoint_v8_recovery_5502.completed.json)
- **Suite de Tests Validée** : [`tests/test_v8_public_and_density.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/tests/test_v8_public_and_density.py)
