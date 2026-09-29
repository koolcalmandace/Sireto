# Standard d'Architecture : Vectorisation Matricielle des Règles (Axe 2)

**Date** : 29 Septembre 2026  
**Branche** : `feature/holding-establishment-reconciliation`  
**Statut** : ✅ **Validé empiriquement sur 200 requêtes réelles (10 000 candidats)**  
**Écart numérique maximal** : **`0.0000000000` (Parité stricte à 100%)**

---

## 1. Contexte & Problématique

Historiquement, l'extraction des features SIRETO s'effectuait de manière scalaire et itérative :
* Pour chaque requête CRM, une boucle parcourait les $N$ candidats ($N \le 500$).
* Pour chaque candidat, 55 règles/features étaient calculées une à une via des dictionnaires Python isolés.
* **Conséquence** : Sur 32 570 requêtes, cela représentait jusqu'à 16 millions d'instanciations de dictionnaires et d'appels unitaires, créant un goulot d'étranglement de plusieurs heures et une fragmentation conceptuelle des signaux.

---

## 2. Décomposition en 4 Blocs Vectoriels Fonctionnels

Les 55 règles de [`FEATURE_NAMES`](../../src/xgb_matcher/features.py) sont désormais regroupées en **4 sections matricielles continues** :

```
┌────────────────────────────────────────────────────────────────────────┐
│                      MATRICE CANDIDATS (N x 55)                        │
├─────────────────────┬───────────────────┬────────────────┬─────────────┤
│ 1. IDENTITÉ         │ 2. SPATIAL        │ 3. ADMIN/LEGAL │ 4. INTERACT │
│ (31 features)       │ (10 features)     │ (5 features)   │ (6 features)│
│ V_text              │ V_geo             │ V_admin        │ V_interact  │
└─────────────────────┴───────────────────┴────────────────┴─────────────┘
```

### A. Bloc 1 : Identité Textuelle ($\mathbf{V}_{\text{text}} \in \mathbb{R}^{N \times 31}$)
* **Rôle** : Empreinte linguistique globale du candidat.
* **Features incluses** : `has_any_name`, `name_count`, `name_jaro_max`, `name_jaro_second`, `name_jaro_gap`, `name_levenshtein_max`, `name_token_overlap_max`, `idf_name`, `numeric_token_match`, `name_first_word_match_max`, `name_contains_crm_max`, `name_crm_contains_cand_max`, `acronym_match_max`, `name_sim_max_etab`, `name_sim_max_ul`, `name_sim_max_sigle`, `name_sim_max_pm_dirigeant`, `type_of_max_name`, `is_ul_name_max`, `is_sigle_max`, `name_length_max`, `has_person_name`, `person_name_jaro_max`, `name_city_overlap_max`, `name_is_city_like_max`, `name_norm_exact`, `alias_match`, `token_overlap_ul`, `ul_vs_pm_indicator`, `name_jaro_vs_enseigne`, `name_city_suffix_match`.

### B. Bloc 2 : Spatial & Géographique ($\mathbf{V}_{\text{geo}} \in \mathbb{R}^{N \times 10}$)
* **Rôle** : Ancrage physique et discrimination territoriale.
* **Features incluses** : `addr_jaro`, `addr_levenshtein`, `postcode_match`, `city_match`, `street_number_diff`, `addr_token_overlap`, `address_density`, `street_name_jaro`, `geo_exact_match`, `street_number_match`.

### C. Bloc 3 : Statut Administratif & Juridique ($\mathbf{V}_{\text{admin}} \in \mathbb{R}^{N \times 5}$)
* **Rôle** : Validation de la conformité juridique et de gouvernance.
* **Features incluses** : `legal_form_category`, `is_siege`, `is_association`, `is_copropriete`, `is_crm_school`.

### D. Bloc 4 : Interactions Bilinéaires ($\mathbf{V}_{\text{interact}} \in \mathbb{R}^{N \times 6}$)
* **Rôle** : Modélisation des croisements multi-critères anti-colocataires et anti-homonymes.
* **Features incluses** : `name_addr_consistency`, `addr_unsupported_by_name`, `name_density_penalty`, `addr_jaro_per_density`, `postcode_match_without_addr`, `full_addr_match_score`.

*(Note : Les 3 features sémantiques neuronales `name_semantic_max`, `name_semantic_second`, `name_semantic_gap` complètent le total à 55 features).*

---

## 3. Validation Empirique sur 200 Requêtes Réelles

Le benchmark comparatif conduit via [`scratch/benchmark_200_matrix_vs_scalar.py`](../../scratch/benchmark_200_matrix_vs_scalar.py) sur 200 requêtes CRM réelles de [`data/crm_ok_gt_merged_v1.csv`](../../data/crm_ok_gt_merged_v1.csv) face au store national [`data/candidates_v8_national`](../../data/candidates_v8_national) établit les métriques suivantes :

| Métrique d'Audit | Valeur Empirique | Interprétation |
| :--- | :---: | :--- |
| **Requêtes CRM évaluées** | **200** / 200 | Couverture 100% |
| **Candidats évalués** | **10 000** | 50 candidats par requête |
| **Points de features vérifiés** | **550 000** | 10 000 $\times$ 55 |
| **Norme L2 moyenne $\mathbf{V}_{\text{text}}$** | **26.7066** | Empreinte nominale stable |
| **Norme L2 moyenne $\mathbf{V}_{\text{geo}}$** | **3 468.3311** | Dominée par les distances et numéros de voirie |
| **Norme L2 moyenne $\mathbf{V}_{\text{admin}}$** | **0.8013** | Sparsité catégorielle maîtrisée |
| **Norme L2 moyenne $\mathbf{V}_{\text{interact}}$** | **0.8678** | Régularisation bilinéaire bornée |
| **Écart maximal absolu ($\max \|X_{\text{mat}} - X_{\text{scal}}\|$)** | **`0.0000000000`** | **Parité mathématique stricte et absolue** |

---

## 4. Garanties Opérationnelles

1. **Invariance des Prédictions** :
   Puisque les matrices d'entrées $X$ sont strictement identiques au bit près, les modèles XGBoost produisent rigoureusement les mêmes scores, les mêmes rangs (Top-1, Top-2, etc.), le même SIRET sélectionné et la même précision finale.
2. **Compatibilité Totale (Zéro Régression)** :
   L'API [`make_features_batch(crm_pre, candidates)`](../../src/xgb_matcher/features.py) est introduite tout en préservant intacte l'interface historique [`make_features_from_preprocessed`](../../src/xgb_matcher/features.py).
