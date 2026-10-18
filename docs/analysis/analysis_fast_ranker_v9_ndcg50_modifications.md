# Rapport Technique : Fast Ranker V9 (NDCG@50, Features Structurelles & Rétention Mono-Occupant)

**Branche Git** : `feature/fast-ranker-v9-ndcg50-retraining`  
**Commits associés** : [`af14b48`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev) (Implémentation V9) et [`eea1ffb`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev) (Documentation handover)  
**Date** : 18 octobre 2026  
**Auteur** : Antigravity (Pair-programming Sireto)

---

## 1. Contexte & Objectif

Lors de l'évaluation nationale sur les **32 570 requêtes CRM** (Étape 6d-nonies) :
- **Retrieval ($K=500$)** : **25 907 cas actifs capturés (98.33%)**. Seuls 439 cas actifs échappent au préfiltre.
- **Fast Ranker ($K=50$)** : **25 219 cas actifs retenus (95.72%)**.
- **Perte Stage 1** : **688 requêtes actives** physiquement présentes dans le pool $K=500$ mais reléguées au-delà du rang 50 par XGBoost.
- **Potentiel prioritaire** : **266 requêtes (38.7% des cas perdus)** sont des *near-misses* immédiats classés entre le rang 51 et 100.

L'objectif de cette branche est de donner à XGBoost les signaux mathématiques et catégoriels naturels, combinés à une fonction de perte LambdaMART orientée Top-50 (`rank:ndcg`), pour repêcher ces ~250-300 cas sans aucun artifice ni dictionnaire statique.

---

## 2. Modifications Implémentées

### A. Fonction de Perte : `rank:pairwise` $\to$ `rank:ndcg` (@50)

- **Fichier modifié** : [`scripts/train_fast_ranker_model.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/train_fast_ranker_model.py)
- **Principe mathématique** :
  - *Pairwise (RankNet)* : pénalise toutes les paires $500 \times 499 / 2 = 124\,750$ paires de manière indifférenciée. Une inversion entre le rang 250 et 251 génère le même gradient qu'entre le rang 49 et 51.
  - *LambdaMART (`rank:ndcg`)* : pondère chaque gradient par le gain marginal NDCG :
    $$\Delta \text{NDCG@50} = \left| \frac{1}{\log_2(\text{rank}_1 + 1)} - \frac{1}{\log_2(\text{rank}_2 + 1)} \right|$$
    Toute inversion au-delà du rang 50 a un poids de gradient nul ($\Delta = 0$). XGBoost concentre 100% de la construction des arbres sur le fait de propulser les candidats pertinents dans le **Top-50**.
- **Paramètres d'entraînement** :
  ```python
  params = {
      "objective": "rank:ndcg",
      "tree_method": "hist",
      "eta": 0.08,
      "max_depth": 6,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "eval_metric": ["ndcg@5", "ndcg@50"],
  }
  ```

---

### B. Nouvelles Features Mathématiques Générales (0 Dictionnaire)

Dans [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py), trois signaux universels ont été créés :

#### 1. `is_commune_authority` (Autorité Publique Territoriale)
- **Fondement** : Nomenclature officielle INSEE. Dans un code commune $C$, il existe **mathématiquement une seule personne morale** de catégorie juridique `7210` (la Commune).
- **Formule** :
  ```python
  cand_cj = str(cand.get("cj_ul") or "").strip()
  is_authority_cj = cand_cj.startswith("72") or cand_cj.startswith("73")
  features["is_commune_authority"] = float(is_authority_cj and features.get("geo_exact_match", 0.0) == 1.0)
  ```
- **Rôle** : Permet à XGBoost d'apprendre que pour les équipements publics locaux (écoles, gymnases, crèches) où le nom de l'équipement diffère du nom légal de la ville, l'institution communale unique du territoire conserve une priorité naturelle.

#### 2. `exact_address_anchor` (Ancrage Physique Strict)
- **Fondement** : Concordance physique exacte entre la Base Adresse Nationale (BAN) et le cadastre.
- **Formule** :
  ```python
  features["exact_address_anchor"] = float(
      features.get("street_number_match", 0.0) == 1.0 and addr_features.get("street_name_jaro", 0.0) >= 0.85
  )
  ```
- **Rôle** : Offre une coupure d'arbre franche empêchant un candidat situé exactement au numéro et dans la rue de la requête d'être relégué en queue par des homonymes de nom éloignés.

#### 3. `acronym_mutual_token_match` (Initialisme Réciproque Générique)
- **Fondement** : Algorithme pur d'extraction des premières lettres sans aucune liste de mots en dur.
- **Formule** :
  - Concaténation des initiales de chaque mot du CRM ($A_{\text{crm}}$) et du candidat ($A_{\text{cand}}$).
  - Validation si $A_{\text{crm}} \in \text{Tokens}_{\text{cand}}$ ou $A_{\text{cand}} \in \text{Tokens}_{\text{crm}}$, ou concordance avec le sigle officiel SIRENE `sigle_ul`.
- **Rôle** : Résout de manière purement textuelle les divergences de type `LNA SANTE` $\leftrightarrow$ `Le Noble Age`.

#### 4. Extension de `FAST_RANKER_FEATURE_SET` à 47 features
- Ajout de : `exact_address_anchor`, `is_commune_authority`, `acronym_mutual_token_match`, `is_association` (INSEE `92xx`), et `full_addr_match_score`.

---

### C. Rétention Guard 5 : Relaxation Ciblée pour Mono-Occupants

Dans [`src/xgb_matcher/retention.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retention.py#L79) :
- **Ajustement** : `mono_occupant_name_jaro_min: float = 0.20` (au lieu de `0.50`).
- **Garde-fou inviolable** :
  - Condition préalable absolue : `address_density <= 1.0`.
  - S'il existe **2 entreprises ou plus** à cette adresse (immeuble collectif, centre d'affaires, zone dense), **Guard 5 est bloqué** (risque de colocataire = 0%).
  - Si et seulement si l'entreprise est la **seule entité existante au numéro et dans la rue dans toute la commune**, le filtre nominal est abaissé pour permettre le repêchage des cas de holdings ou d'enseignes commerciales divergentes.

---

### D. Échelle d'Entraînement 10K (GroupKFold SIREN)

- **Volume d'entraînement** : 10 000 requêtes nationales (au lieu de 6 398 dans le V8).
- **Validation croisée** : `GroupKFold(n_splits=3, groups=df["siren"])` garantissant zéro fuite entre train et dev.
- **Résilience** : Persistance atomique par chunks compressés de 500 requêtes (`.npz` + `manifest.json`), tolérance complète aux interruptions avec auto-resume instantané.
- **Modèle de sortie** : [`models/xgbranker_fast_v9.json`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/models/xgbranker_fast_v9.json).

---

## 3. Matrice de Traçabilité & Tests

| Composant | Fichier source | Statut des Tests |
|---|---|---|
| Features V9 | [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) | ✅ Validé (`test_v9_fast_ranker_features`) |
| Guard 5 Mono-Occupant | [`src/xgb_matcher/retention.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retention.py) | ✅ Validé (`test_guard5_mono_occupant_address_rescue`) |
| Suite de non-régression | `tests/` (122 tests au total) | ✅ **122 / 122 Passés (0 échec, 0 régression)** |
| Script d'apprentissage | [`scripts/train_fast_ranker_model.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/train_fast_ranker_model.py) | 🔄 En cours d'exécution (10 000 requêtes) |
