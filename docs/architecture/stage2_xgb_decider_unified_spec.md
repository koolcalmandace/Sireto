# Spécification d'Architecture : Décideur Probabiliste Stage 2 Unifié & Éradication des Discordances de Scores

**Date** : 19 octobre 2026  
**Statut** : En attente d'approbation (Standby for Approval)  
**Branche Git** : `feature/fast-ranker-v9-ndcg50-retraining`  
**Auteurs** : Équipe Antigravity & Lead Architecture SIRETO  
**Artefacts Cibles** :  
- Modèle Stage 2 : `models/xgb_decider_v9_unified.json`
- Calibrateur Probabiliste : `models/xgb_decider_v9_calibrator.pkl`
- Schéma de Features : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) (`FEATURE_NAMES`, 68 features unifiées)
- Moteur d'Inférence 2-Stages : [`src/xgb_matcher/infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py)

---

## 1. Contexte & Rôle Métier du Stage 2 dans le Pipe V7

Dans l'architecture Pipe V7 de SIRETO, le système de rapprochement opère selon une chaîne déterministe à 3 étages :

```
CRM Input (Nom, Adresse, CP, Ville)
   │
   ▼
[Stage 1 : Fast Ranker V9] ──► FILTRAGE RAPIDE & RÉTENTION (Top-500 -> Top-50).
   │                            - Modèle LambdaMART (rank:ndcg @50), 47 features.
   │                            - Rôle BINAIRE : 100% data-driven, sélectionne les 50 meilleurs.
   ▼
[Stage 2 : Décideur XGBoost]──► DÉCISION PROBABILISTE INDIVIDUELLE CALIBRÉE.
   │                            - Modèle XGBClassifier (binary:logistic) + Calibrateur Isotonique.
   │                            - RÉSCORE L'INTÉGRALITÉ DES 50 CANDIDATS avec les 68 features complètes.
   │                            - Calcule la probabilité réelle P(Match = 1 | X) ∈ [0, 1].
   ▼
[Stage 3 : Routing Risk Model]► ANALYSE DE LA SCÈNE COMPLÈTE & DÉCISION D'AUTOMATISATION.
   │                            - Analyse le score_top1, score_top2, le score_gap (marge).
   │                            - Décide : AUTO (Match direct) ou REVIEW (orienté Places).
   ▼
Sortie MATCH ou Fallback Google Places ("Places as CRM Repair")
```

### Contrat d'Interface du Stage 2 :
1. **Indépendance d'ordre** : Le Décideur ne se fie pas aveuglément au rang du Stage 1 ; il réévalue les 50 candidats sur une base probabiliste absolue.
2. **Sortie Probabiliste Certifiée** : Chaque candidat reçoit une probabilité calibrée. Un score de $0.90$ doit signifier mathématiquement que sur 100 cas similaires, exactement 90 sont des vrais positifs.
3. **Génération de la Scène pour le Stage 3** : Le Décideur extrait les métriques clés de la requête :
   - `top1_score` : probabilité du meilleur candidat.
   - `top2_score` : probabilité du second candidat.
   - `score_gap` : marge de certitude ($\text{top1\_score} - \text{top2\_score}$).
   - `score_ratio` : ratio de dominance ($\text{top1\_score} / \text{top2\_score}$).

---

## 2. Le Problème de Discordance de Scores ("Stage Inversion") & Principe d'Inclusion Stricte

### 2.1 Qu'est-ce que la Discordance de Scores ?
Lorsque deux modèles successifs ont des schémas de features désynchronisés, un phénomène pervers d'inversion apparaît :
1. Le **Stage 1 (Fast Ranker V9)** a identifié un établissement au **Rang 1 direct** grâce aux nouvelles features continues BAN (distance GPS $< 50\text{m}$, `exact_address_anchor = 1.0`).
2. Le **Stage 2 (Décideur)** arrive avec ses embeddings sémantiques. Si le modèle Stage 2 n'a pas été entraîné avec ces mêmes features BAN, il ignore la perfection géographique du candidat. S'il détecte une légère variation d'enseigne, il peut rétrograder ce bon candidat au profit d'un homonyme sémantique situé à 30 km.

### 2.2 La Règle d'Architecture : Inclusion Stricte ($F_{\text{Stage 1}} \subset F_{\text{Stage 2}}$)
Pour interdire mathématiquement toute régression ou désaveu arbitraire du Stage 1 par le Stage 2 :

$$\mathbf{F}_{\text{Stage 2}} = \mathbf{F}_{\text{Stage 1}} \cup \{\text{Embeddings Sémantiques lourds}\} \cup \{\text{Projections NAF Vectorielles}\} \cup \{\text{Noms Civils}\}$$

Toutes les 47 features qui ont permis au Fast Ranker V9 d'atteindre 96.25% de rappel Top-50 **sont intégralement transmises au Décideur**.

### 2.3 Comment Fusionner Sémantique et Géographie sans Conflit ?

1. **Arbres d'Interactions Profonds (`max_depth = 7`)** :
   Dans XGBoost, les features ne s'additionnent pas de manière linéaire : elles se conditionnent par embranchements successifs :
   - *Branche Concordante* : **SI** `exact_address_anchor == 1.0` (adresse BAN exacte) **ET** `name_semantic_max >= 0.70` $\rightarrow$ **Probabilité $P = 0.99$**.
   - *Branche Homonyme* : **SI** `name_semantic_max == 0.95` (noms quasi identiques) **MAIS** `spatial_distance_km > 10.0` $\rightarrow$ **Probabilité $P = 0.05$ (Rejet immédiat)**.
2. **Porte de Garde Lexicale & Spatiale (`semantic_gate_allows`)** :
   Déjà implémentée dans [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) : la similarité sémantique est mise à zéro si le candidat ne possède pas une ancre minimale lexicale ou spatiale.
3. **Injection du Signal Stage 1 (`stage1_score` et `stage1_rank`)** :
   Le Décideur reçoit le score brut et le rang issu du Stage 1 comme features supplémentaires, lui fournissant une ancre de stabilité amont.

---

## 3. Les 5 Leviers d'Amélioration du Stage 2

### Levier 1 : Protocole d'Entraînement Étanche sur les 32K (`GroupKFold(siren)`)
Conformément aux Master Rules de gouvernance du projet, **on n'entraîne jamais sur l'intégralité du dataset 32K** :

```
                              DÉCOUPAGE STRICT GROUP-KFOLD(SIREN)
┌──────────────────────────────────────┬───────────────────────────┬───────────────────────────┐
│              TRAIN (70%)             │          DEV (15%)        │         TEST (15%)        │
│          ~22 800 requêtes            │      ~4 900 requêtes      │      ~4 900 requêtes      │
├──────────────────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Apprentissage des arbres XGBoost     │ Validation continue       │ Évaluation aveugle finale │
│ et descente de gradient binaire      │ Early Stopping (30 rounds)│ Certification officielle  │
│ (scale_pos_weight = 100)             │ Calibration Isotonique    │ Zero Data Leakage garanti │
└──────────────────────────────────────┴───────────────────────────┴───────────────────────────┘
```

- **Garantie d'Étanchéité** : Le découpage est effectué par `GroupKFold(siren)`. Si une entreprise (ex: `TotalEnergies` ou un cabinet médical) possède 15 établissements sur toute la France, la totalité de ses SIRETs est confinée dans un seul et même fold.
- **Filtrage Qualité** : Les requêtes sans Ground Truth certain sont exclues du jeu d'entraînement.

---

### Levier 2 : Intégration Continue des Projections NAF Vectorielles (Levier A)
- **Le Cas d'Usage** : Départager les colocataires en adresse dense (ex: un cabinet dentaire, une pharmacie et une agence immobilière au même numéro de rue).
- **Implémentation** : Exploiter le moteur vectoriel continu [`src/xgb_matcher/naf_vector_engine.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/naf_vector_engine.py) (732 classes d'activités INSEE).
- **Feature `naf_vector_similarity`** : Calcul continu de la similarité cosinus entre le texte d'activité/raison sociale CRM et la définition vectorielle de la classe APE de chaque candidat SIRENE.
- **Impact** : Discrimination data-driven immédiate sans aucune liste manuelle de mots-clés.

---

### Levier 3 : Prise en Compte Native des Noms Civils Inversés (Levier B - EI `CJ 1000`)
- **Le Cas d'Usage** : 70.3% des Entreprises Individuelles (médecins, artisans, professions libérales) sont enregistrées en `NOM PRENOM` dans SIRENE (`nom_ul`, `prenom_usuel_ul`), alors que le CRM saisit `Prénom Nom` ou `Cabinet Dr Nom Prénom`.
- **Implémentation dans `features.py`** :
  ```python
  # Double permutation continue native
  civil_p_n = f"{prenom_usuel} {nom}"
  civil_n_p = f"{nom} {prenom_usuel}"
  civil_name_jaro = max(jaro(crm_name, civil_p_n), jaro(crm_name, civil_n_p))
  ```
- **Sécurité** : Pour les personnes morales (SARL, SAS), la feature prend la valeur neutre par défaut ($0.0$). L'arbre active cette branche conditionnellement au code juridique `cj_ul.startswith('10')`.

---

### Levier 4 : Arbitrage Établissement d'Exploitation vs Siège Social
- **Le Cas d'Usage** : Dans le Stage 1, la **Dual-Preservation** garantit que le Siège social et l'établissement local d'une structure multi-sites entrent tous deux dans le Top-50.
- **Implémentation** : Fournir au Décideur la feature différentielle d'adresse :
  $$\text{addr\_concordance\_delta\_vs\_siege} = \text{street\_jaro}(\text{Candidat Local}, \text{CRM}) - \text{street\_jaro}(\text{Siège}, \text{CRM})$$
- Si le CRM a explicitement demandé l'adresse de proximité, le delta est largement positif ($+1.0$), et le Décideur recommande l'établissement local. Si le CRM n'a spécifié aucune rue ou a saisi l'adresse du siège, le Décideur privilégie le siège social institutionnel.

---

### Levier 5 : Accentuation Naturelle de la Marge & Sanctuarisation de la Calibration

> [!CAUTION]
> **Avertissement Formel de Rigueur Scientifique** :  
> Il est formellement interdit d'appliquer tout forçage artificiel (multiplicateurs post-hoc, seuils rigides manuels) pour écarter le `score_gap`. Cela corromprait la distribution des probabilités et ferait chuter la précision en production.

#### Comment la marge s'élargit-elle de manière 100% Organique ?
1. **Grâce aux features discriminantes (NAF vectoriel, Noms civils, Adresses BAN exactes)** :
   - Au lieu d'hésiter entre deux colocataires avec des scores indécis de $0.55$ et $0.52$ ($\text{gap} = 0.03$), l'arbre affecte naturellement **$0.94$** au vrai établissement et **$0.06$** au colocataire ($\text{gap} = \mathbf{0.88}$).
   - La marge devient immense et nette de manière purement data-driven.
2. **Validation par le Brier Score & Diagramme de Fiabilité (Reliability Curve)** :
   - Le modèle entraîné doit satisfaire un Brier Score minimal ($\le 0.04$) et une calibration isotonique stricte.
3. **Conséquence directe sur le Stage 3 (Routing Risk Model)** :
   - Disposant d'un `score_gap` beaucoup plus tranché, le Risk Model pourra faire passer le taux d'automatisation de **74.12% à plus de 80-82% en mode AUTO**, tout en sanctuarisant la précision absolue de **99.84%**.

---

## 4. Plan de Validation & Statut

* **Statut Actuel** : Document de spécification finalisé. **En attente d'approbation utilisateur (Standby for Approval)** avant tout lancement d'entraînement ou modification de script.
* **Prochaines Étapes Opérationnelles post-approbation** :
  1. Préparation du jeu d'échantillons Top-50 Fast Ranker V9 partitionné par `GroupKFold(siren)` (70% Train, 15% Dev, 15% Test).
  2. Entraînement contrôlé de `xgb_decider_v9` avec calibration isotonique sous PowerGuard.
  3. Mesure différentielle certifiée des métriques sur le Test Set indépendant.
