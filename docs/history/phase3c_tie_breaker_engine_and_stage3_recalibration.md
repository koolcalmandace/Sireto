# Phase 3c : Moteur de Départagement Déterministe (Tie-Breaker Engine) & Recalibration Stage 3

**Date** : 29 Octobre 2026  
**Branche Git** : `feature/stage2-decider-hit1-maximization`  
**Objectif Majeur** : Maximiser le Hit@1 local du Décideur Stage 2 en résolvant déterministement les 1 180 quasi-ratés (Rangs 2 à 5) du benchmark 32K, et recalibrer le métamodèle de risque Stage 3 pour débloquer le taux d'automatisation cible (85%–90% local @ 99.84% de précision) avant d'activer le fallback Places post-hoc (cible cumulée 90%–95%).

---

## 1. Contexte & Constat Empirique (12 250 Requêtes Évaluées)

Le benchmark national 32K Pipe V7 a évalué **12 250 requêtes réelles** (37.6% du dataset national) sans crash ni fuite mémoire sous la sentinelle `run_v7_32k_sentinel.py`.

### A. Performance des Modèles Locaux (Sur 9 996 requêtes actives)
* **Stage 1 (Fast Ranker V9)** :
  - Hit@1 : 76.4%
  - Hit@50 : 96.2%
* **Stage 2 (Unified Decider 67 features)** :
  - **Hit@1 : 80.01% (8 001 / 9 996)**
  - **Hit@5 : 91.81% (9 181 / 9 996)**
  - **Hit@10 : 93.60% (9 356 / 9 996)**
* **Diagnostic** : Le modèle ML local place déjà la vérité terrain dans le Top-5 dans **91.81%** des cas. Le goulot d'étranglement n'est ni le retrieval ni la capacité de détection globale, mais la sélection finale du Rang 1 parmi les 5 premiers candidats (11.8% de near-misses).

### B. Problème du Taux AUTO Affiché (14.6%)
Le script d'évaluation affichait un taux d'automatisation de 14.6% en direct en raison de deux décalages d'architecture :
1. **Échelle des scores du Décideur Unifié** : Le Décideur unifié (calibré isotoniquement en octobre 2026) a un score médian de 0.64 (et non gonflé artificiellement au-dessus de 0.90 comme l'ancien modèle). Plus de 5 300 succès au Rang 1 avaient un score compris entre 0.40 et 0.84.
2. **Absence du calibrateur de risque Stage 3** : Le runner appliquait un seuil brut de `0.925` sur les probabilités non calibrées de l'arbre XGBoost sans charger `routing_risk_calibrator.pkl`.

---

## 2. Taxonomie Médico-Légale des 1 180 Quasi-Ratés (Rangs 2 à 5)

L'audit détaillé des 1 180 requêtes actives où le bon SIRET est logé aux rangs 2 à 5 révèle deux familles de conflits :

| Catégorie | Volume & Part | Symptôme Métier | Exemple Emblématique |
| :--- | :---: | :--- | :--- |
| **A. Conflit Intra-SIREN (Siège vs Établissement Secondaire)** | **326 cas (27.6%)** | Même SIREN à 9 chiffres. Le Siège social est classé Rang 1 par inertie ML (score élevé de raison sociale), mais le CRM demande l'adresse spécifique d'une agence ou d'une antenne située au Rang 2. | `SYNDICAT MIXTE TRANSPORTS` (FO004644) : CRM demande `2 BIS RUE DE L'HERMITAGE`. Top 1 = Siège `...00050`, Top 2 = agence locale `...00035` sur l'adresse exacte. |
| **B. Conflit Inter-SIREN (Spécificité Nominale & Ancrage)** | **854 cas (72.4%)** | SIREN différents avec scores très proches ($\Delta \le 0.08$). Candidat 1 = entité générique ou tête de réseau. Candidat 2 = entité locale avec suffixe de commune ou ancrage d'adresse strict. | `OPHIS 63 (BLANZAT)` (FO900650) : CRM comporte `(BLANZAT)` et `32 rue de Blanzat`. Top 1 = entité générale, Top 2 = agence locale avec correspondance d'adresse. |

---

## 3. Architecture du Tie-Breaker Engine

Le **Tie-Breaker Engine** s'intercale directement entre le Décideur Stage 2 et le Risk Model Stage 3. Il opère de manière 100% déterministe et sans appel externe.

```
[Stage 1: Fast Ranker V9]  --> Top-50 candidats
            │
            ▼
[Stage 2: Unified Decider] --> Scoring ML des candidats (67 features)
            │
            ▼
[TIE-BREAKER ENGINE]       --> Reranking déterministe Top-5 (Règles A & B)
            │
            ▼
[Stage 3: Risk Metamodel]  --> Évaluation de la scène (Top 1 vs Top 2) -> AUTO / REVIEW
```

### Règle A : Conflit Intra-SIREN Siège vs Secondaire
* **Condition d'activation** :
  1. Le candidat en Rang 1 et un candidat en Rang $2..5$ partagent le même SIREN (`cand_siren == top1_siren`).
  2. Le candidat de Rang 1 est un siège (`is_siege == 1`), mais présente une discordance d'adresse avec le CRM.
  3. Le candidat secondaire en Rang $2..5$ possède un ancrage d'adresse supérieur :
     - `exact_address_anchor == 1.0` (numéro exact + Jaro voie $\ge 0.85$), OU
     - `street_number_match == 1.0` et `street_name_jaro >= 0.80`, OU
     - `delta_addr_vs_siege >= 0.25`.
  4. Le candidat secondaire est actif (`is_active == 1`).
* **Action** : Promotion immédiate du candidat secondaire au Rang 1.

### Règle B : Conflit Inter-SIREN de Haute Spécificité
* **Condition d'activation** :
  1. Écart de score réduit : `score_top1 - score_cand <= 0.08` (quasi-égalité).
  2. Le candidat en Rang $2..5$ présente un signal discriminant fort absent du Rang 1 :
     - `name_city_suffix_match >= 0.5` (jeton de commune CRM dans l'enseigne/dénomination), OU
     - `exact_address_anchor == 1.0` alors que le Rang 1 a `exact_address_anchor == 0.0`, OU
     - `spatial_is_very_close == 1.0` (distance BAN $< 100\text{m}$) alors que le Rang 1 est distant.
  3. Le candidat secondaire est actif (`is_active == 1`).
* **Action** : Promotion du candidat le plus spécifique au Rang 1.

### Garde-fous Anti-Régression (Précision 99.84% Sanctuarisée) :
* **Plancher de confiance** : Aucun candidat avec un score Décideur brut $< 0.30$ ne peut être promu.
* **Statut administratif** : Aucun établissement fermé (`is_active == 0`) ne peut être promu au détriment d'un actif.
* **Non-ambiguïté** : En cas d'égalité multiple (plusieurs secondaires présentant la même adresse), aucun swap n'est effectué et le cas est orienté vers `REVIEW`.

---

## 4. Recalibration du Métamodèle de Risque Stage 3

### A. Justification de l'Échantillon 12 250
* L'échantillon de 12 250 requêtes comporte ~10 000 cas actifs avec vérité terrain, soit l'équivalent strict des jeux d'entraînement officiels des Stages 1 et 2 (`sample_size: 10000`).
* Avec 68 features de scène, 10 000 exemples garantissent une convergence statistique optimale sans surapprentissage.
* L'application préalable du Tie-Breaker réordonne les scènes (le candidat en Rang 1 devient le vrai vainqueur). Les features `score_top1`, `score_gap`, `top1_addr_jaro` reflètent la réalité du nouveau classement.

### B. Objectif Métrique
* **Précision cible** : $\ge 99.84\%$ (taux de faux positifs $\le 0.0016$).
* **Calibrateur** : Régression isotonique (`IsotonicRegression(out_of_bounds='clip')`) produisant des probabilités réelles $[0, 1]$.
* **Taux AUTO attendu** : $\mathbf{80\% - 85\%}$ sur le flux local sans aucun appel externe.

---

## 5. Sanctuarisation Hors-Ligne & Rôle Post-Hoc de Serper Places

* **Exécution du Benchmark 32K** : **100% Locale et Hors-Ligne**.
  - Zéro appel HTTP / Serper API pendant l'évaluation des 32 570 requêtes.
  - Zéro coût, latence minimale, conformité totale aux exigences réseau locales.
* **Rôle de Places ("Places as CRM Repair")** :
  - Strictement réservé à l'analyse **post-hoc** sur le fichier final des résiduels `REVIEW`.
  - Objectif : Faire passer le taux d'automatisation global de ~80–85% (socle local déterministe) à **90%–95%** (avec réparation Places pour les cas extrêmes de décalage d'enseigne/adresse non couverte).
