# Rapport d'Analyse : Étape 1 - Pré-Retrieval & Stockage Partitionné

**Date** : 20 Août 2026  
**Auteur** : Antigravity & User  
**Fichier analysé** : [`src/xgb_matcher/partitioned_store.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/partitioned_store.py)  
**Artefacts associés** : `data/candidates_v7_all/insee/`, `data/candidates_v7_all/cp/`, `manifest/insee_counts.parquet`

---

## 1. Rôle et Objectif de l'Étape 1
L'Étape 1 constitue la couche d'accès aux données (I/O, partitionnement, projection de colonnes et cache mémoire).
Sa mission est de fournir à l'Étape 2, en temps constant ($O(1)$ à $O(\text{taille partition})$) et avec un minimum d'empreinte RAM, la totalité des établissements bruts correspondant à la localisation géographique de la requête CRM.

---

## 2. Fonctionnalités et Mécanismes Clés

### A. Structure Hive Parquet String-Safe
- **Partitionnement double** :
  - Par commune : `<partitions_dir>/insee/insee=XXXXX/*.parquet`
  - Par code postal (fallback) : `<partitions_dir>/cp/postcode=XXXXX/*.parquet`
- **Typage strict** : Forçage des identifiants (`siret`, `siren`, `insee`, `postcode`) en chaînes de caractères (évite la troncature des zéros initiaux comme `06000` ou `01000`).

### B. Manifeste d'Indexation $O(1)$ (`manifest/insee_counts.parquet`)
- Charge en amont la table de correspondance `(insee, row_count)`.
- Permet de connaître immédiatement le nombre d'établissements dans une commune avant même d'ouvrir le dataset Parquet.

### C. Politique des Mega-Communes (`full_insee`)
- **Seuil** : 100 000 établissements (ex: Paris, Lyon, Marseille).
- **Principe** : Charge l'intégralité de la commune en un seul bloc pour neutraliser les erreurs d'arrondissement fréquentes dans les CRM (ex: un CRM indiquant 75001 pour un établissement situé dans le 75008).

### D. Optimisation de la Bande Passante et de la RAM (`REQUIRED_COLUMNS`)
- Seules les colonnes consommées par le feature engineering et la conformité sont lues depuis le disque :
  - Identifiants : `siret`, `siren`, `insee`, `postcode`, `city`.
  - Noms & Enseignes : `denomination`, `enseigne1`, `enseigne2`, `enseigne3`, `denomination_ul`, `denomination_usuelle_ul`, `sigle_ul`, `nom_ul`, `prenom_usuel_ul`, `pm_dirigeant_names`.
  - Voies : `numeroVoie`, `typeVoie`, `libelleVoie`, `complementAdresse`.
  - Statut : `is_siege`, `etablissementSiege`, `etat_admin`, `cj_ul`, `last_treatment_date`.
- Les colonnes volumineuses ou inutilisées (`nom_usage_ul`, `pseudonyme_ul`, etc.) sont exclues dès la lecture PyArrow.

### E. Cache Mémoire LRU
- Maintien en RAM d'un cache `OrderedDict` des 5 dernières partitions INSEE et CP chargées (`_MAX_STORE_CACHE_SIZE = 5`).
- Accélère drastiquement les traitements par lots lorsque des requêtes CRM consécutives appartiennent à la même commune.

---

## 3. Synthèse de la Frontière Étape 1 vs Étape 2

| Caractéristique | Étape 1 (`partitioned_store.py`) | Étape 2 (`retrieval.py`) |
| :--- | :--- | :--- |
| **Responsabilité** | I/O, chargement Parquet, cache LRU. | Filtrage, scoring TF-IDF, whitelists, expansion. |
| **Entrée** | Code INSEE ou Code Postal. | Données brutes de commune issues de l'Étape 1 + Ligne CRM. |
| **Sortie** | Tous les établissements de la commune (500 à 100k). | Pool d'élite qualifié de 100 à 500 candidats. |

---

## 4. Conclusion
L'Étape 1 est **validée et conforme**. Elle assure une lecture sans perte, rapide et maîtrisée en mémoire.
