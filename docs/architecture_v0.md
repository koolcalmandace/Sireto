# SIRETO Architecture Baseline (V0)
## Guide Complet de l'Architecture de Bout en Bout : Du Pré-Retrieval au Post-Decider

**Version** : V0 (Baseline SSOT)  
**Date** : 20 Août 2026  
**Objectif Métier** : Réconciliation déterministe haute performance CRM $\leftrightarrow$ SIRENE/SIRET sans LLM (Cible : $\ge 74.5\%$ AUTO @ $99.84\%$ de précision).

---

## 1. Schéma Fonctionnel Global (End-to-End)

```mermaid
flowchart TD
    subgraph E0["0. Ingestion & Pre-Processing"]
        A[CSV CRM Brut<br/>Nom, Adresse, CP, INSEE] --> B[preprocess_crm_row<br/>features.py & naming.py<br/>Normalisation texte, regex voies, clés géo]
    end

    subgraph E1["1. Pre-Retrieval & Stockage Partitionné"]
        B --> C[PartitionedCandidateStore<br/>data/candidates_v7_all/]
        C --> D[(Hive Partitions<br/>insee=XXXXX / cp=XXXXX)]
        D --> E[Lecture Lazy + Manifeste O1<br/>Gestion Mega-Communes full_insee]
    end

    subgraph E2["2. Retrieval & Pool Construction"]
        E --> F[Filtrage Métier<br/>drop_unnamed / include_closed]
        F --> G[Indexation Locale TF-IDF<br/>Noms 1-2 words + Chars 3-5 + Voies]
        F --> H[Whitelists de Sauvetage Rescue<br/>addr_hash MD5 + numeric_tokens]
        G & H --> I[Prefilter Hybride Top-500<br/>Sparse TF-IDF + Dense FAISS]
        I --> J[Étape V8b: Expansion SIREN<br/>Injection frères via siren_to_geo.parquet]
        J --> K[Pool Final de Candidats<br/>100 à 500 établissements]
    end

    subgraph E3["3. Feature Engineering"]
        K --> L[features.py: Extraction 68 Features<br/>- Lexicales: Jaro, Levenshtein, IDF, overlap<br/>- Géo: num_diff, street_jaro, density<br/>- Typologie: siege, association, nature juridique<br/>- Interactions V8: colocataires, homonymes]
    end

    subgraph E4["4. Stage 1: Fast Ranker"]
        L --> M[XGBoost Booster FAST<br/>models/xgbranker_fast_*.json<br/>rank:ndcg / rank:pairwise<br/>0 calcul sémantique]
        M --> N[Top-50 Candidats Qualifiés]
    end

    subgraph E5["5. Stage 2: Semantic Decider"]
        N --> O[Calcul Batch BERT / CamemBERT<br/>Similarités cosinus sous Semantic Gate]
        O --> P[XGBoost Decider Champion<br/>models/xgb_decider_*.json<br/>Score fin des 50 candidats]
        P --> Q[Extraction Top-1 & Top-2<br/>Candidat élu + Challenger direct]
    end

    subgraph E6["6. Stage 3: Post-Decider Risk Metamodel"]
        Q --> R[Construction Scène 68 Features<br/>score_top1, score_top2, score_gap, deltas]
        R --> S[Risk Model XGBoost + Calibrateur Isotonique<br/>models/routing_risk_model.pkl<br/>Estimation probabilité de succès]
        S --> T{Risk Score >= 0.835 ?}
    end

    subgraph E7["7. Post-Decider Fallback: Places as CRM Repair"]
        T -->|Oui| U[Statut AUTO<br/>MATCH SIRET Top-1 direct]
        T -->|Non| V[Statut REVIEW<br/>Requête Serper Google Places]
        V --> W{Dept-Guard<br/>CP Places[:2] == CP CRM[:2]}
        W -->|Non / Invalide| X[NO_MATCH]
        W -->|Oui| Y[Création CRM Réparé<br/>Nom + Adresse Google Places]
        Y --> Z[Rerun XGBoost Complet<br/>Stage 1 -> Stage 2 -> Stage 3]
        Z --> AA{Re-run Status == AUTO ?}
        AA -->|Oui| AB[MATCH_PLACES<br/>SIRET Validé via Places]
        AA -->|Non| X
    end

    subgraph E8["8. Export & Audit"]
        U & AB & X --> AC[Export Final CSV/JSON<br/>SIRET, SIREN, Provenance, Logs]
    end
```

---

## 2. Description Détaillée des 8 Étapes

---

### Étape 0 : Ingestion CRM & Pré-traitement Déterministe
- **Fichiers sources** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py) et [`src/xgb_matcher/naming.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/naming.py)
- **Objectif** : Nettoyer, normaliser et structurer la ligne CRM d'entrée.
- **Détail des opérations** :
  1. **Normalisation textuelle** : Conversion majuscule, suppression des accents et de la ponctuation (`normalize_text`, `normalize_name`).
  2. **Découpage de l'adresse** : Extraction par expressions régulières du numéro de voie (`crm_street_num`) et du corps de la voie (`crm_street_name`).
  3. **Formatage des identifiants géographiques** :
     - Code Postal normalisé sur 5 chiffres.
     - Code INSEE normalisé sur 5 caractères.

---

### Étape 1 : Pré-Retrieval & Stockage Partitionné
- **Fichier source** : [`src/xgb_matcher/partitioned_store.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/partitioned_store.py)
- **Objectif** : Accès ultra-rapide aux 14 millions d'établissements de la base SIRENE nationale.
- **Détail des opérations** :
  - **Structure Hive Parquet** : Stockage partitionné dans `data/candidates_v7_all/insee/insee=XXXXX/` et `data/candidates_v7_all/cp/postcode=XXXXX/`.
  - **Manifeste $O(1)$** : `manifest/insee_counts.parquet` permet de connaître la volumétrie d'une commune instantanément.
  - **Politique Mega-Communes** : Pour les communes dépassant 100 000 lignes (Paris, Lyon, Marseille), chargement en mode `full_insee` pour éliminer les erreurs de saisie d'arrondissement CRM.
  - **Gestion de la mémoire** : Cache LRU (`_cache_insee`, `_cache_cp`) évitant les I/O répétées pour les requêtes de la même zone géographique.

---

### Étape 2 : Retrieval & Construction du Pool de Candidats
- **Fichiers sources** : [`src/xgb_matcher/retrieval.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retrieval.py), [`src/xgb_matcher/blocking.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/blocking.py), [`src/xgb_matcher/tfidf_cache.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/tfidf_cache.py)
- **Objectif** : Réduire le volume d'une commune (jusqu'à 100 000 lignes) à un pool d'élite de 100 à 500 candidats avec un **Recall $\ge 99\%$**.
- **Détail des opérations** :
  1. **Filtrage métier** : Élimination des coquilles sans nom (`drop_unnamed`), conservation des fermés (`include_closed=True`) pour prévenir les faux positifs.
  2. **Indexation TF-IDF locale (avec cache persistant)** :
     - *Noms* : TF-IDF mots n-grammes (1,2) sur dénominations, enseignes, sigles.
     - *Caractères* : TF-IDF caractères n-grammes (3,5) (*char_wb*) pour rattraper acronymes et fautes d'orthographe.
     - *Adresses* : TF-IDF n-grammes (1,2) sur les voies.
  3. **Whitelists de Sauvetage Universel (Rescue)** :
     - `addr_hash` : Hash MD5 (numéro + voie) repêchant tous les voisins d'immeuble.
     - `numeric_tokens` : Repêchage sur les numéros présents dans le nom (ex: "Pharmacie du 8 Mai", "Bus 42").
  4. **Pré-filtre Hybride** : Union dédupliquée des 500 meilleurs par nom + 500 par adresse + Top FAISS dense (si activé) + Whitelist de sauvetage.
  5. **Étape V8b (Expansion SIREN)** :
     - Consultation du mapping `siren_to_geo.parquet` pour chaque SIREN du pré-filtre.
     - Injection ordonnée des établissements frères locaux et cross-partitions (actifs > fermés, sièges > secondaires).

---

### Étape 3 : Feature Engineering Complet
- **Fichier source** : [`src/xgb_matcher/features.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/features.py)
- **Objectif** : Représenter chaque paire (CRM, Candidat SIRENE) par un vecteur de **68 features explicatives**.
- **Familles de features** :
  - **Lexicales / Noms (28 features)** : `name_jaro_max`, `name_jaro_gap`, `name_levenshtein_max`, `name_token_overlap_max`, `idf_name`, `numeric_token_match`, `name_first_word_match_max`, `name_contains_crm_max`, `name_crm_contains_cand_max`, `acronym_match_max`, `name_sim_max_etab`, `name_sim_max_ul`, `name_sim_max_sigle`, `name_sim_max_pm_dirigeant`, etc.
  - **Adresses / Géographie (10 features)** : `addr_jaro`, `addr_levenshtein`, `postcode_match`, `city_match`, `street_number_diff`, `addr_token_overlap`, `address_density`, `street_name_jaro`, `name_addr_consistency`, `street_number_match`.
  - **Typologie & Statut (6 features)** : `is_siege`, `is_association`, `legal_form_category` (PUBLIC / PRIVE / INCONNU), `is_crm_school`, `alias_match`, `token_overlap_ul`.
  - **Interactions V8 (7 features anti-colocataires & homonymes)** :
    - `addr_unsupported_by_name` : $\text{addr\_jaro} \times (1 - \text{name\_jaro\_max})$.
    - `name_density_penalty` : $\text{density} \times (1 - \text{name\_jaro\_max})$.
    - `addr_jaro_per_density` : $\text{addr\_jaro} / \ln(1 + \text{density})$.
    - `postcode_match_without_addr` : $\text{postcode\_match} \times (1 - \text{street\_name\_jaro})$.
    - `full_addr_match_score` : $0.5 \times \text{street\_name\_jaro} + 0.3 \times \text{street\_number\_match} + 0.2 \times \text{postcode\_match}$.
    - `name_jaro_vs_enseigne` : Similarité directe avec l'enseigne de l'établissement.
    - `name_city_suffix_match` : Ratio de tokens de la commune présents dans l'enseigne.

---

### Étape 4 : Stage 1 - Fast Ranker (Élagage Rapide)
- **Fichiers sources** : [`src/xgb_matcher/infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py#L410), [`scripts/train_xgb_ranker.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/train_xgb_ranker.py)
- **Objectif** : Élaguer 100 à 500 candidats pour extraire les **50 meilleurs** en < 5ms.
- **Modèle** : `models/xgbranker_fast_*.json` (XGBoost `rank:ndcg` / `rank:pairwise`).
- **Particularité** : Exécute uniquement les features non-sémantiques pour garantir une latence minimale.

---

### Étape 5 : Stage 2 - Semantic Decider (Re-Ranking & Élection Top-1)
- **Fichiers sources** : [`src/xgb_matcher/infer.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/infer.py#L428), [`src/xgb_matcher/semantic.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/semantic.py), [`scripts/train_xgb_decider.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/train_xgb_decider.py)
- **Objectif** : Désigner avec une précision maximale le candidat **Top-1** parmi les 50 finalistes.
- **Détail des opérations** :
  1. **Semantic Gating** : Vérification que la proximité lexicale minimale est atteinte avant d'engager le calcul sémantique.
  2. **Calcul BERT Vectoriel** : Similarité cosinus CamemBERT calculée par batch (`name_semantic_max`, `name_semantic_second`, `name_semantic_gap`).
  3. **Scoring XGBoost Decider** : Application du modèle champion `models/xgb_decider_*.json`.
  4. **Extraction** : Identification du **Top-1** (meilleur score) et du **Top-2** (challenger direct).

---

### Étape 6 : Post-Decider - Stage 3 Risk Metamodel & Routing
- **Fichiers sources** : [`src/xgb_matcher/routing_risk.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/routing_risk.py), [`scripts/route_xgb_results.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/route_xgb_results.py), [`scripts/train_routing_risk_model.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/train_routing_risk_model.py)
- **Objectif** : Séparer "trouver le candidat en tête" de **"être absolument certain de la décision"**.
- **Scène de Décision (68 features)** :
  - Métriques d'écart : `score_top1`, `score_top2`, $\text{score\_gap} = \text{score\_top1} - \text{score\_top2}$, $\text{score\_ratio} = \frac{\text{score\_top1}}{\text{score\_top2}}$.
  - Deltas de features pour chaque variable $F$ : $\Delta F = \text{top1\_}F - \text{top2\_}F$.
- **Règle de décision** :
  - Modèle XGBoost Isotonique `models/routing_risk_model.pkl`.
  - Si $\text{Risk Score} \ge 0.835 \implies$ **`AUTO`** (Match validé directement avec le SIRET Top-1, $99.84\%$ de précision).
  - Si $\text{Risk Score} < 0.835 \implies$ **`REVIEW`** (Aiguillage vers le fallback Places).

---

### Étape 7 : Post-Decider Fallback - "Places as CRM Repair"
- **Fichiers sources** : [`src/pipe_v6/places_fallback.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/pipe_v6/places_fallback.py), [`src/pipe_v6/serper_places_client.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/pipe_v6/serper_places_client.py)
- **Objectif** : Upgrader les cas `REVIEW` en `MATCH` sans créer de faux positif.
- **Principe fondamental** : *Google connaît l'identité commerciale de l'entreprise, XGBoost identifie le SIRET légal.*
- **Déroulement** :
  1. **Requête Google Places** : Appel API Serper avec le nom et l'adresse CRM.
  2. **Dept-Guard** : Rejet immédiat si $\text{CP}_{\text{Places}}[:2] \neq \text{CP}_{\text{CRM}}[:2] \implies$ **`NO_MATCH`**.
  3. **CRM Réparé** : Reconstruction d'un objet CRM propre à partir des données Places (nom officiel de l'établissement, adresse validée).
  4. **Re-run XGBoost Complet** : Ré-exécution exacte du pipeline complet (Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 3).
  5. **Sortie Binaire Post-Places** :
     - Si statut ré-exécuté $== \text{AUTO} \implies$ **`MATCH_PLACES`**.
     - Sinon $\implies$ **`NO_MATCH`** (Aucun statut `REVIEW` résiduel après Places).

---

### Étape 8 : Export & Télémetrie Finale
- **Fichiers sources** : [`src/pipe_v6/exporter.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/pipe_v6/exporter.py), [`scripts/evaluate_routing.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/scripts/evaluate_routing.py)
- **Restitution** : Fichier final CSV / Parquet contenant :
  - Identifiants légaux : `siret`, `siren`.
  - Décision finale : `AUTO`, `MATCH_PLACES`, ou `NO_MATCH`.
  - Scores de confiance et logs explicatifs d'auditabilité complète.
