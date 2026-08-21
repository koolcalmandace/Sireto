# Phase 3 : Résolution Géo Stricte & Expansion CRM (32k)

**Période** : Août 2026 (En cours)  
**Branche Git** : `feature/geo-resolution-and-crm-expansion`  
**Objectif Majeur** : Résoudre proprement la hiérarchie géographique (INSEE $\rightarrow$ découverte CP $\rightarrow$ vide sans fallback départemental), auditer et fusionner l'incrément CRM (32 570 requêtes), et valider l'absence de régression.

---

## 1. Chantier 1 : Résolution Géographique Déterministe
* **Problématique** : Certaines requêtes CRM n'ont pas de code INSEE direct mais un code postal, risquant un chargement massif et bruité de tout le bucket CP.
* **Solution Implémentée** :
  - Méthode `load_with_geo_resolution()` et `_discover_insee_codes_from_cp()` dans `src/xgb_matcher/partitioned_store.py`.
  - Hiérarchie stricte :
    1. Si INSEE présent $\rightarrow$ chargement direct de la partition communale (`load_by_insee`).
    2. Si INSEE absent $\rightarrow$ découverte des codes INSEE enfants partageant ce code postal, puis chargement ciblé par commune.
    3. Si aucun résultat $\rightarrow$ retour pool vide et log explicite `GEO_RESOLUTION_EMPTY` (zéro fallback départemental).
  - Branchement direct dans `build_candidate_pool()` (`src/xgb_matcher/retrieval.py`) et `src/xgb_matcher/infer.py` pour un zéro train/serve skew absolu.

---

## 2. Chantier 2 : Audit & Fusion Contrôlée du Dataset CRM
* **Audit Qualité de l'Incrément (15 516 requêtes UNSEEN)** :
  - Zéro champ obligatoire vide (`crm_name`, `crm_adresse`, `crm_cp`).
  - 100% de nouveaux SIRENs (zéro fuite / zéro chevauchement avec les 14 198 SIRENs de la base 17k).
  - Validation de fusion accordée ([`audit_crm_increment_20260817.md`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/docs/analysis/audit_crm_increment_20260817.md)).
* **Script de Fusion Contrôlée** :
  - `scripts/merge_crm_increment.py` pour générer `data/crm_ok_gt_merged_v1.csv` (32 570 requêtes).

---

## 3. Feuille de Route d'Exécution & Validation
1. **Étape 1 (Validée)** : Câblage et unification de la résolution géo dans le pipeline partagé.
2. **Étape 2 (En cours)** : Benchmark empirique sur les 17 054 cas baseline (`run_geo_benchmark_17k.bat`).
3. **Étape 3 (Suivante)** : Benchmark empirique sur les 15 516 cas de l'incrément (`run_geo_benchmark_increment.bat`).
4. **Étape 4 (Suivante)** : Exécution de la fusion contrôlée (32 570 cas).
5. **Étape 5 (Suivante)** : Audit architectural pas-à-pas (Étapes 2 à 7) avant le réentraînement des modèles.

---

## 4. Commits Majeurs Associés
* `c754b97` : Résolution géo stricte dans `partitioned_store.py` et script de fusion CRM.
* `fc4396b` : Unification de `build_candidate_pool` dans `retrieval.py` et script d'évaluation `evaluate_geo_resolution.py`.
