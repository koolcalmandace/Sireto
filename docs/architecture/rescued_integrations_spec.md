# Spécification Officielle des Améliorations Intégrées (Pipe V7 Hybride)

**Date d'intégration** : 22 Septembre 2026  
**Branche** : `feature/semantic-vector-sidecar-and-naf-projection`  
**Statut** : Validé et testé (38/38 tests unitaires au vert)  
**Périmètre** : Documentation officielle de l'intégration dans le dépôt principal des composants sains issus des audits de *Project North Star*, exempts de toute surcompensation ou règle métier surspécifique.

---

## 🏛️ Vue d'Ensemble de l'Architecture Implémentée

Les composants intégrés s'articulent harmonieusement avec le socle XGBoost 3-Stages et le **Sidecar Sémantique** :

```mermaid
flowchart TD
    A[Requête CRM Brute] --> B[FONCTION 1 : preprocess_crm_row<br/>(Voies BAN + Délimiteurs + Numéros de voirie multi-positions & plages)]
    
    B --> C[STAGE 1 : Retrieval Hybride Local<br/>(build_candidate_pool)]
    C --> D[(Base Locale INSEE / CP)]
    C --> E[SIDECAR SÉMANTIQUE<br/>(Piliers 1 & 2 : Whitelist Marques & Acronymes)]
    
    D & E --> F[Pool Candidats Unifié & Dédupliqué<br/>(Top-500 Préfiltre TF-IDF + Whitelist Rescue)]
    
    F --> G[FONCTION 2 : make_features_from_preprocessed<br/>(Densité adresse + Copro 9110 + Jaccard + Priorité Voie + Acronymes pairs)]
    G --> H[SIDECAR SÉMANTIQUE : Pilier 3<br/>(Projection Vectorielle NAF 732 classes)]
    
    H --> I[STAGE 2 : Scoring ML XGBoost<br/>(Ranker Top-50 + Decider Binaire)]
    I --> J{STAGE 3 : Risk Metamodel v2.0<br/>(Modèle de Risque 68 Features)}
    
    J -->|AUTO| K[MATCH Automatique @ 99.84% Précision]
    J -->|REVIEW| L[Étage LLM Expert Arbitrage]
```

---

## 🛠️ Détail des Composants Intégrés dans `src/xgb_matcher/`

### 1. Correction de Symétrie des Acronymes (Bug Fix C5)
* **Fichier** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) (`acronym_match`)
* **Problème résolu** : Le code précédent testait si l'une des chaînes était l'acronyme de l'autre, mais omettait la branche où les deux chaînes sont des acronymes. Par conséquent, deux acronymes strictement identiques (`SNCF` vs `SNCF`, `RATP` vs `RATP`, `CCAS` vs `CCAS`) renvoyaient par erreur `0`.
* **Correction appliquée** :
  ```python
  norm_a = a.replace(" ", "").upper()
  norm_b = b.replace(" ", "").upper()
  a_is_acronym = _is_acronym(norm_a)
  b_is_acronym = _is_acronym(norm_b)

  # Both are acronyms: exact match
  if a_is_acronym and b_is_acronym:
      return int(norm_a == norm_b)
  ```

---

### 2. Normalisation BAN des Types de Voies
* **Fichier** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) (`normalize_street_type`)
* **Problème résolu** : Les requêtes CRM abrègent couramment les types de voies (`12 av de la Paix`, `4 bd Haussmann`), provoquant une chute artificielle de la similarité Jaro face aux libellés SIRENE / BAN qui utilisent les formes complètes (`AVENUE`, `BOULEVARD`).
* **Correction appliquée** :
  - Table fermée des abréviations officielles de la Base Adresse Nationale (BAN) : `AV` $\rightarrow$ `AVENUE`, `BD`/`BLVD` $\rightarrow$ `BOULEVARD`, `R` $\rightarrow$ `RUE`, `RTE` $\rightarrow$ `ROUTE`, `IMP` $\rightarrow$ `IMPASSE`, `ALL` $\rightarrow$ `ALLEE`, `CHE` $\rightarrow$ `CHEMIN`, `ZA`/`ZI`/`ZAC` $\rightarrow$ zones économiques.
  - Confiné **strictement au champ adresse** dans `preprocess_crm_row()` avec frontières de mots regex (`\b`) pour protéger les mots réguliers (`AVRIL`, `HAVRE`).

---

### 3. Multi-Représentation sur Délimiteurs Structurels
* **Fichier** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) (`preprocess_crm_row` et `make_features_from_preprocessed`)
* **Problème résolu** : Dans les CRM, les dénominations d'associations ou de succursales comportent souvent des compléments de service (*« UNAPEI - Foyer Robinson »*, *« CROIX ROUGE / SECTION JEUNES »*). Tronquer l'entrée détruit l'information ; ne pas la séparer dégrade le score Jaro face à la raison sociale juridique enregistrée (*« UNAPEI »*).
* **Correction appliquée** :
  - Extraction de `crm_name_root` sur les séparateurs structurels (` - `, ` / `) sur la chaîne brute avant substitution des tirets.
  - Dans la boucle de comparaison des candidats, le moteur évalue conjointement `crm_name` et `crm_name_root` et retient le maximum de similarité (`sim_jaro`, `sim_lev`, `token_overlap`), garantissant un appariement optimal sans perte de contexte.

---

### 4. Parser de Plages de Voirie `[66, 68]` et Numéro Multi-Positions Sécurisé
* **Fichier** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) (`extract_street_number`, `parse_street_numbers`, `street_number_diff`)
* **Problème résolu** : 
  1. Les parcelles cadastrales multi-numéros (*« 66-68 Avenue des Champs Élysées »*) étaient tronquées au premier chiffre (`66`), créant un écart d'adresse si SIRENE était au `68`.
  2. Les adresses CRM inversées (*« Boulevard Foch 14 »*) ou précédées de compléments (*« Bâtiment B, 14 rue de Paris »*, *« BP 42, 5 rue du Commerce »*) échouaient sur l'ancrage strict `^(\d+)`.
* **Correction appliquée** :
  - Détection native des plages `(\d+)\s*[-/]\s*(\d+)` renvoyant `"min-max"`.
  - Nettoyage amont des préfixes d'immeuble et postaux (`BP`, `Bâtiment`, `Étage`, `Local`) avant capture du numéro de voie multi-positions.
  - Fonction `parse_street_numbers()` extrayant l'ensemble des entiers : si le candidat SIRENE est au 66 OU au 68, `street_number_match = 1.0` et `street_number_diff = 0.0`.

---

### 5. Garde-Fou Copropriété Continue (`cj_ul == '9110'`)
* **Fichier** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py)
* **Problème résolu** : Les syndicats de copropriétaires portent le nom et l'adresse physique de l'immeuble et viennent parasiter les requêtes visant des entreprises situées à cette adresse. Les exclusions dures d'antan éliminaient à tort les requêtes cherchant légitimement un syndic.
* **Correction appliquée** :
  - Création de la feature continue : `features["is_copropriete"] = float(str(cand.get("cj_ul") or "").strip() == "9110")`.
  - Enregistrée dans `FEATURE_NAMES` pour apprentissage ML sans filtre rigide.

---

### 6. Injection Sécurisée du Sidecar Sémantique dans le Retrieval (Stage 1)
* **Fichiers** : [`src/xgb_matcher/retrieval_config.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval_config.py) & [`retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py)
* **Problème résolu** : Le préfiltre TF-IDF élimine au rang >500 les enseignes commerciales disjointes de leur dénomination légale (`LIDL` vs `SNC BANQUE COMMERCIALE`) ou les sigles d'organismes publics.
* **Correction appliquée** :
  - Ajout du flag `rescue_sidecar: bool = True` dans `RetrievalConfigV1`.
  - Intégration dans l'étape de rescue whitelist de `retrieval.py` : appel à `run_entity_sidecar(crm_name, crm_address, pool_list)` et injection directe des SIRETs repêchés dans `whitelisted_sirets`.

---

## 🧪 Validation & Couverture de Tests

Une suite de tests unitaires dédiée a été créée dans [`tests/test_rescued_enhancements.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/tests/test_rescued_enhancements.py).

Résultats de l'exécution complète du banc de test :
```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Kabouassi\.gemini\antigravity-ide\scratch\Sireto-dev
collected 38 items

tests\test_naf_vector_engine.py ........................                 [ 63%]
tests\test_entity_sidecar.py .........                                   [ 86%]
tests\test_rescued_enhancements.py .....                                 [100%]

============================= 38 passed in 5.86s ==============================
```

**100% des 38 tests unitaires passent avec succès en 5.86 secondes.**
