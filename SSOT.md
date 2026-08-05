# SIRETO Routing SSOT (Single Source of Truth)

**Date** : 30 juillet 2026  
**Performance cible** : **36.8% AUTO @ 81.8% Precision (Version 4.0 Calibrated)**

## 1. Architecture du Pipeline (Pipe V7.1 - Version 4.0)

Le pipeline utilise une architecture XGBoost à 3 stages remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML.

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1 (Fast Ranker)** | `xgbranker_fast_20260730_155642.json` | Sélectionne 50 candidats SIRENE sans sémantique (ML Pruning rapide). Modèle `rank:ndcg`. |
| **Stage 2 (Semantic Re-Ranker)** | `xgbranker_20260730_155642.json` | Re-trie les 50 candidats avec les features sémantiques BERT pour extraire le "Top-1". Modèle `rank:ndcg`. |
| **Stage 3 (Risk Calibrator)** | `routing_risk_model.pkl` | Juge le candidat Top-1 pour décider si la requête est AUTO ou REVIEW. Modèle `binary:logistic` Isotonic XGBoost. |

## 2. Partitions candidats (point de départ)

Les partitions SIRENE sont la source canonique des candidats et le point de départ du filtrage.

| Élément | Valeur | Rationale |
|---------|--------|-----------|
| **Chemin** | `data/candidates_v7_all/` | Stock candidats partitionné par commune (V7 string-safe) |
| **Partitionnement** | `insee/` et `cp/` (Hive) | Accès rapide par commune ou CP |
| **Génération** | `scripts/build_candidate_partitions_v5.py` | Pipeline unique de build (Types String forcés) |
| **Nettoyage** | Suppression des partitions existantes avant build | Évite l'accumulation de doublons SIRET |
| **Contraintes** | Pas de doublons SIRET dans une partition | Garantie pour TF-IDF + ranker |
| **Mega-communes** | Seuil 100 000 lignes | Policy `full_insee` pour coverage maximal |
| **Scarcity Gate** | `len(pool_list) < 100` | Gâte les Stream 1 & 2 fallbacks pour éviter les fuites de mémoire (MemoryError) dans les grandes villes |

## 3. Stratégie de Retrieval "Ultima" (Stage 1)

Le retrieval est conçu pour un recall quasi-total sans fallback départemental inutile.

Standard unique : **Variant B** (Bag-of-names + char fallback).

| Paramètre | Valeur | Rationale |
|-----------|--------|-----------|
| **Pool Mode** | `insee_then_postcode` | Strict commune, fallback CP local |
| **TF-IDF Name** | `bag` + word ngrams (1,2) | Aligné sur l'entraînement v5fast (token_pattern \b\w+\b) |
| **TF-IDF Addr** | word ngrams (1,2), norm=None | Repêche par adresse, robuste aux variantes de voie |
| **Char TF-IDF** | ngrams (3,5) | Fallback acronymes/typos sur le nom |
| **Rescue** | `addr_hash` + `numeric` | Whitelist systématique (matchs exacts) |
| **Prefilter k** | 500 | Union des canaux Nom + Adresse |
| **Stage 1 top-N** | 50 | Envoyés au Stage 2 (Decider) |
| **Rescue post-ranker** | Aucun | Pas d'ajout de candidats hors top-N |

## 4. Stratégie d'Entraînement (Confirmed 250 Subset)

L'alignement Train/Serve est garanti par l'usage des mêmes modules de retrieval.

- **Phase 0 : Retrieval (aligné Train/Serve)**
  - Pool strict `insee_then_postcode` + double TF-IDF (nom + adresse) + rescue universel.
  - `prefilter_k=500`, padding deterministe si pool < min_candidates.
  - **Mega-policy** : `full_insee` pour supprimer les pertes dues au CP CRM.
- **Phase 1 : Ranker FAST (Stage 1 / No Semantic)**
  - Entraîné sur le pool de retrieval (ULTIMA B).
  - Sampling "Turbo" : 50 négatifs choisis via rangs TF-IDF du retrieval.
  - Objectif : maximiser le Recall@50 (`rank:ndcg`).
- **Phase 2 : Decider (Stage 2 / Semantic Enabled)**
  - Entraîné sur le pool de 50 candidats du Fast Ranker.
  - Semantic feature computation activé par défaut.
  - Entraîné sur le sous-ensemble de 250 exemples confirmés de haute qualité pour éliminer le bruit des faux positifs de V3.5.
- **Phase 3 : Calibrator (Stage 3 / Risk Model)**
  - Isotonic regression recalibrée sur les scores de validation pour garantir une probabilité non biaisée.

## 5. Artefacts Canoniques (Version 4.0)

| Artefact | Chemin | Description |
|----------|--------|-------------|
| **Ranker (Fast)** | `models/xgbranker_fast_20260730_155642.json` | Stage 1 Champion (Filtrage rapide) |
| **Ranker (Semantic)** | `models/xgbranker_20260730_155642.json` | Stage 2 Champion (Re-ranking avec BERT) |
| **Meta** | `models/xgb_two_stage_meta_20260730_155642.json` | Meta Ranker aligné SSOT |
| **GT Data** | `data/crm_ok_gt.csv` | Gold Standard (corrigé INSEE/CP) |

## 6. Environnement d'execution (SSOT)

Les developpements et optimisations doivent respecter la contrainte materielle suivante :

- Machine cible : Windows 11 Core i7 / MacBook Pro M4 Pro (latence et memoire doivent rester compatibles)

---
*Note : Ce document est la source unique de vérité. Toute modification doit respecter le principe de "Zero Skew".*
