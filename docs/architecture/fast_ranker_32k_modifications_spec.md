# Spécification Technique : Modifications & Rétention du Fast Ranker (Stage 1) sur le Benchmark 32k

**Date** : 2 octobre 2026  
**Branche Git Dédiée** : `experiment/fast-ranker-32k-retention` (isolée depuis `0add3fb`)  
**Objectif** : Maximiser la rétention des candidats du pool Retrieval ($K=500$, plafond certifié de **98.16%** de rappel sur les actifs) vers le Top-50 ($K=50$) du Fast Ranker sur l'intégralité du dataset national (32 884 requêtes CRM, ~26 738 actifs).

---

## 1. Contexte & Problématique

Le pipeline V7 repose sur une cascade à 3 étages :
1. **Retrieval Géographique & Élargi ($K=500$)** : Pré-sélection géolocalisée (INSEE / CP / expansion SIREN / sauvetages déterministes d'adresses et sidecar). Il atteint un rappel certifié de **98.16%** sur les établissements actifs au niveau national.
2. **Stage 1 : Fast Ranker ($K=500 \to K=50$)** : Modèle XGBoost léger (features lexicales, géographiques et administratives rapides, $< 2.0\text{ ms}$ / requête, 0 calcul d'embeddings sémantiques lourds). Son rôle est d'élaguer $90\%$ des candidats superflus tout en transmettant $\ge 99.5\%$ des vrais positifs au Stage 2.
3. **Stage 2 : Decider & Reranker ($K=50 \to K=5$)** : Modèle complet avec features sémantiques fines et interaction multi-établissements.
4. **Stage 3 : Risk Metamodel & Fallback Places** : Arbitrage AUTO vs REVIEW et réparation des déconnexions d'enseignes par Google Places.

### La Perte Constatée (Baseline initiale sur 3k)
Sur l'échantillon représentatif de 3 000 requêtes (756 actives testées avec GT dans le pool), le Fast Ranker initial n'atteignait que **96.16% de Hit@50** (727 / 756), soit une perte de **29 établissements actifs** entre le Retrieval et le Fast Ranker.

---

## 2. Analyse Médico-Légale (Forensic Root Causes)

L'audit approfondi mené sur les 29 cas perdus a révélé trois causes racines structurelles et une frontière métier incompressible :

### Cause 1 : Empoisonnement du Sidecar (`_is_sidecar_match`)
- **Fichier** : `src/xgb_matcher/retrieval.py` (lignes 780-785)
- **Mécanisme défaillant** : La liste `whitelisted_sirets` regroupait indifféremment :
  - Les sauvetages par hash d'adresse (`addr_index`)
  - Les sauvetages par jetons numériques (`num_index`)
  - L'expansion du secteur public / municipalités
  - Les véritables ancres sémantiques Sidecar (`sidecar_sirets`)
  Le code appliquait ensuite aveuglément :
  ```python
  if whitelisted_sirets:
      for c in candidates:
          if c.get("siret") in whitelisted_sirets:
              c["_is_sidecar_match"] = 1.0  # <-- ERREUR MAJEURE
  ```
- **Conséquence** : **481 081 candidats non-GT** (39.8% du pool !) recevaient à tort le flag `sidecar_match = 1.0`. Dans les immeubles professionnels multi-sociétés (ex. 10 colocataires au même numéro de rue), des cabinets comptables, dentistes et notaires recevaient un boost de $+1.5$ à $+2.3$ points XGBoost, saturant les rangs 1 à 10 et reléguant le véritable établissement CRM au-delà du rang 50.

### Cause 2 : Omission de `is_siege` dans les Features Fast Ranker
- **Fichiers** : `src/xgb_matcher/features.py`
- **Mécanisme défaillant** : Bien que calculée dans `FEATURE_NAMES` (ligne 166), la variable `is_siege` était omise de `FAST_RANKER_FEATURE_SET` (ligne 193).
- **Conséquence** : Pour les entités multi-sites (communes, hôpitaux, grandes entreprises avec 30 à 100 annexes dans la même commune), les annexes locales partageant l'adresse CRM monopolisaient les 50 premières places, tandis que le Siège Social (souvent attendu par le CRM ou porteur de la raison sociale) était classé au rang 52–67.

### Cause 3 : Absence de Pénalisation des Colocataires purs (`addr_unsupported_by_name`)
- **Fichiers** : `src/xgb_matcher/features.py`
- **Mécanisme** : La feature `addr_unsupported_by_name` ($= \text{addr\_jaro} \times (1.0 - \text{name\_jaro\_max})$) était réservée au Stage 2.
- **Conséquence** : Le Fast Ranker ne pouvait pas pénaliser un candidat ayant une adresse strictement identique ($\text{addr\_jaro} = 1.0$) mais un nom sans aucun rapport ($\text{name\_jaro} = 0.15$), laissant les colocataires dépasser le vrai candidat.

### Cause 4 : Frontière Métier Incompressible (Rebranding / Enseigne absente de SIRENE)
- Plusieurs cas analysés (ex. `UICRA` $\to$ `FRANCE CHIMIE AUVERGNE RHONE ALPES`, `LES 4 ECLUSES` $\to$ `ASS ARTS SCENIQUES ROCKS`) présentent **0 jeton lexical commun** dans la base officielle SIRENE.
- **Règle architecturale** : Le Fast Ranker étant purement lexical et déterministe, il est **voulu et normal** que ces cas sans correspondance lexicale ne soient pas devinés au Stage 1. C'est précisément le rôle attribué dans le Pipe V7 au **Stage 3 Risk Model $\to$ Fallback Places API ("Places as CRM Repair")**.

---

## 3. Détail des Modifications Appliquées

Les 4 modifications retenues sont strictement ciblées, déterministes et non-polluantes :

### Modification 1 : Décontamination de `_is_sidecar_match`
- **Fichier** : `src/xgb_matcher/retrieval.py`
- **Action** : Isoler `sidecar_sirets` de `whitelisted_sirets`. Seuls les véritables candidats découverts par le module Sidecar reçoivent `c["_is_sidecar_match"] = 1.0`. Les candidats issus du hash d'adresse ou des jetons numériques entrent dans le pool pour assurer le rappel géographique sans recevoir ce boost artificiel indu.

### Modification 2 : Ajout de `is_siege` et `addr_unsupported_by_name` dans `FAST_RANKER_FEATURE_SET`
- **Fichier** : `src/xgb_matcher/features.py`
- **Action** :
  ```python
  FAST_RANKER_FEATURE_SET: Set[str] = {
      ...
      "is_siege",                  # Rétablissement du signal siège social
      "addr_unsupported_by_name",  # Pénalisation des colocataires sans support lexical
  }
  ```
  Le nombre de features passe de 36 à 38, restant strictement dans le budget de calcul rapide ($< 1.8\text{ ms}$).

### Modification 3 : Règle de Rétention Top-50 (Protection Siège & Sidecar)
- **Fichier** : Pipeline d'arbitrage / sélection du Top-50.
- **Action** :
  - Les $K=50$ candidats finaux sont principalement triés par le score XGBoost décroissant.
  - **Garantie de rétention** : Si un candidat possède `is_siege == 1.0` pour un SIREN déjà présent dans le Top-50, ou s'il s'agit d'une ancre identifiée par le Sidecar (`sidecar_match == 1.0`), il est protégé dans le budget des 50 candidats transmis au Stage 2.
- **Résultat prouvé sur prototype** : Rétention portée de **96.16% à 98.81%** sur les actifs avec GT dans le pool (pertes réduites de 29 à seulement 9 requêtes).

---

## 4. Protocole d'Évaluation sur les 32k Requêtes Nationales

### Dataset
- **Fichier** : `data/crm_ok_gt_merged_v1.csv`
- **Volume** : 32 884 requêtes CRM annotées (dont ~26 738 actives 'A' et ~6 146 fermées 'F').
- **Partitions SIRENE** : `data/candidates_v8_national` (96 départements consolidés).

### Architecture du Batch (Safe Batch Standard)
- **RAM sous contrôle** : `MemoryGovernor` avec seuil cible à 3.0 Go et plafond critique à 3.8 Go (vidage dynamique des caches Parquet et LRU).
- **Localité géographique** : Tri préalable des requêtes par `(insee, cp)` afin de maximiser le taux de hit LRU Parquet (>99% des requêtes consécutives sur la même commune réutilisent le cache en mémoire).
- **Tolérance aux pannes & Veille** : Checkpointage atomique sur disque par blocs de 250 requêtes. Reprise automatique transparente (`--resume`). Détection PowerGuard pour mise en veille sûre.
- **Sorties produites** :
  - `reports/fast_ranker_32k/fast_ranker_32k_summary.json`
  - `reports/fast_ranker_32k/fast_ranker_32k_details.csv` (diagnostic ligne à ligne avec rang exact Stage 1 et cause d'échec le cas échéant).

### Métriques Clés Évaluées
- **Hit@1, Hit@5, Hit@10, Hit@20, Hit@50** global et segmenté par état administratif (Actifs 'A' vs Fermés 'F').
- **Taux de Rétention K=500 $\to$ K=50** : Comparaison directe avec le plafond certifié de 98.16% du Retrieval.
