# Phase 2 : Exploration Route B (SIREN-First) & Pivot vers V8b (SIREN Expansion)

**Période** : Février - Mars 2026  
**Objectif Majeur** : Évaluer l'approche hiérarchique au niveau SIREN global, identifier ses limites, et stabiliser l'architecture V8b (V7 + expansion SIREN post-prefilter).

---

## 1. Contexte & Travaux sur la Route B (SIREN-First Global)
* **Hypothèse initiale** : Résoudre l'entité juridique (SIREN) au niveau national via un index TF-IDF global, puis filtrer géographiquement les établissements SIRET de cette entité.
* **Développements réalisés** :
  - Module `src/xgb_matcher/siren_retrieval.py` et constructeur d'index `scripts/build_siren_global_index.py`.
  - Mapping national `siren_to_geo.parquet` associant chaque SIREN à ses localisations.
* **Constat & Limites** :
  - Sensibilité accrue aux erreurs de saisie sur noms d'entreprises génériques ou acronymes courts par rapport au filtrage local strict.
  - Risque d'élimination prématurée au Stage 1 global.
* **Décision** : Maintenir la Route B dans le code pour expérimentation A/B offline, mais abandonner son statut de default de production.

---

## 2. Pivot vers la Trajectoire V8b (Chemin Retenu)
* **Principe de V8b** :
  1. Conserver le prefilter local V7 (INSEE / CP) comme seed de base robuste.
  2. Ajouter une étape d'**expansion SIREN (Step 5)** : à partir des SIRENs identifiés dans le pré-filtre, injecter les établissements frères locaux et cross-partitions via `siren_to_geo.parquet`.
  3. Enrichir le feature engineering avec **7 features d'interaction V8** pour discriminer les colocataires et homonymes géographiques.
* **Bénéfices** : Robustesse locale maximale combinée au rattrapage des établissements multi-sites et des transferts de siège.

---

## 3. Commits Structurants Associés
* `35fb441` : Features d'interaction V8, hard negatives colocataires/siblings et hyperparamètres decider.
* `3e090b7` : Implémentation Route B et index SIREN global.
* `9c0e806` / `f1fbbb8` / `c961371` : Implémentation de l'expansion SIREN V8b et découplage geo-only.
