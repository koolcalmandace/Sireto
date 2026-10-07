# Audit Médico-légal des Pertes Intermédiaires Fast Ranker 32K (10 240 Requêtes)

**Date** : 7 Octobre 2026  
**Échantillon analysé** : 10 240 requêtes évaluées et persistées dans [`reports/fast_ranker_32k/fast_ranker_32k_v8_retention_details.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/fast_ranker_32k/fast_ranker_32k_v8_retention_details.csv)  
**Modèle** : `xgbranker_fast_v8.json` (38 features rapides sans sémantique lourde)  
**Budget de filtrage** : $K=500 \longrightarrow K=50$  

---

## 1. 📊 Bilan Chiffré Global & Taux de Rétention

Sur les 10 240 premières requêtes du dataset national de 32 570 :

| Catégorie | Requêtes | GT dans Pool $K=500$ | Hit@1 | Hit@5 | Hit@10 | Hit@20 | **Hit@50 (Budget)** | **Taux de Rétention ($K=500 \to 50$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Établissements Actifs ('A')** | **8 291** | **8 106 (97.77%)** | 5 935 (71.58%) | 7 198 (86.82%) | 7 458 (89.95%) | 7 634 (92.08%) | **7 877 (95.01%)** | **97.17%** *(Perte ranker: 2.83%)* |
| **Établissements Fermés ('F')** | 1 949 | 1 657 (85.02%) | 767 (39.35%) | 1 396 (71.63%) | 1 472 (75.53%) | 1 524 (78.19%) | **1 574 (80.76%)** | **94.99%** *(Perte ranker: 5.01%)* |
| **ENSEMBLE (Actifs + Fermés)** | **10 240** | **9 763 (95.34%)** | 6 702 (65.45%) | 8 594 (83.93%) | 8 930 (87.21%) | 9 158 (89.43%) | **9 451 (92.29%)** | **96.80%** *(Perte ranker: 3.20%)* |

> [!NOTE]
> Le Fast Ranker accomplit son objectif principal : **diviser la taille du pool par 10 (de 500 à 50)** tout en conservant **97.17% des candidats actifs légitimes**. La perte nette imputable au modèle d'arbres de décision n'est que de **229 cas sur 8 291 (2.76%)**.

---

## 2. 🔍 Décomposition des Pertes sur les Établissements Actifs

Sur les 8 291 requêtes actives, **414 requêtes (4.99%)** ne sont pas dans le Top-50 final :
- **185 requêtes (2.23%) — Perte Stage 0 (Hors Pool $K=500$)** :
  - **0 commune manquante** : 100% des requêtes ont une partition Parquet valide (`pool_size > 0`).
  - Le Ground Truth a été éliminé dès le prefilter TF-IDF / Sidecar en amont car la chaîne de recherche CRM et le nom SIRENE divergent trop radicalement.
- **229 requêtes (2.76%) — Perte Stage 1 (Fast Ranker, rang $> 50$)** :
  - Le Ground Truth était bien présent dans les 500 candidats.
  - Le Fast Ranker l'a classé au-delà de la 50ème position.

---

## 3. 🧬 Taxonomie Médico-légale des 229 Pertes Fast Ranker

L'analyse unitaire croisée avec les tables SIRENE nationales et le fichier CRM révèle 6 causes distinctes :

| Famille | Nb Cas | Part | Description & Mécanisme de Perte | Cas Emblématiques |
| :--- | :---: | :---: | :--- | :--- |
| **1. Collectivités & Équipements Publics** | **75** | **32.8%** | Le CRM désigne un équipement public local (école, collège, gymnase, espace d'animation, service de l'eau) alors que le GT SIRET est rattaché à la personne morale globale (Commune, Conseil Départemental, Agglomération). | • `ESPACE LOUIS PARANT` (rg 1684) $\to$ `CA GRAND BOURG AGGLOMERATION`<br>• `EPE DAUDET` (rg 1876) $\to$ `CA GRAND BOURG AGGLOMERATION`<br>• `DEPARTEMENT DE LAISNE` (rg 1370) $\to$ `COLLEGE JEAN MERMOZ`<br>• `CONVERGENCE-LAON` (rg 222) $\to$ `COMMUNE DE LAON` |
| **2. Divergence Contrat / Facturation CRM** | **72** | **31.4%** | Le CRM enregistre le nom d'un marché public / projet informatique interne (`CONVERGENCE-LAON INFORMATIQUE PUBLI`), mais le SIRET facturé est l'organisme HLM ou la mairie. Zéro corrélation lexicale. | • `CONVERGENCE-LAON INFORMATIQUE PUBLI` (48 occurrences identiques dans le CRM) mappé sur `OFFICE PUBLIC HABITAT AISNE` (`OPAL`) |
| **3. Centres d'affaires & Domiciliation Dense** | **60** | **26.2%** | Plusieurs dizaines d'entreprises partagent exactement la même adresse (centres de domiciliation d'Antibes, Cap 3000 à Nice, ZAC). Des colocataires passifs submergeaient le Top-50. | • `PRESTIGE IMEX AUTOS` (rg 73) au 55 av de Cannes à Antibes<br>• `Opticien Nice Cap 3000 GrandOptical` (rg 64) au Centre Commercial Cap 3000<br>• `HOTEL LE G` (rg 231) $\to$ `DEVINTINA` |
| **4. Personnes Physiques (Agents, Artisans)** | **15** | **6.6%** | Professionnels indépendants où SIRENE n'a pas de raison sociale (`denomination == None`), mais uniquement `nom_ul` et `prenom_usuel_ul`. Le CRM porte l'enseigne nationale du réseau. | • `GROUPAMA SUPPORTS ET SERVICES` (rg 66) $\to$ `CHAZALMARTIN PASCAL` (agent général Groupama)<br>• `Maison Monticelli` (rg 150) $\to$ `JARNIAC SANDRA`<br>• `GROUPAMA` (rg 560) $\to$ `FAYRET REGIS` |
| **5. Enseignes & Franchises Déconnectées** | **2** | **0.9%** | Société d'exploitation franchisée ayant une dénomination juridique sans rapport avec la marque commerciale de façade. | • `JD Sports` (rg 221) $\to$ `JD`<br>• `BRIT HOTEL CAEN NORD-MEMORIAL` (rg 60) $\to$ `RELAIS DU BOCAGE` |
| **6. Near Misses Stricts (Rangs 51–70)** | **2** | **0.9%** | Forte similarité lexicale, mais forte densité locale qui repousse le candidat juste au seuil d'exclusion. | • `DR FOAHOM` (rg 55) $\to$ `SCM DES DOCTEURS FOAHOM KAMWA - PHAN`<br>• `MAISON DE L'HABITAT` (rg 58) $\to$ `SETE THAU HABITAT OPH` |
| **7. Erreurs GT pures / Inexpliquées** | **3** | **1.3%** | Données corrompues ou sans lien démontrable. | Cas résiduels. |

> [!IMPORTANT]
> **Constat Majeur** : Plus de **64% des pertes (147 / 229)** résultent de désignations de facturation contractuelle (`CONVERGENCE-LAON`, équipements communaux/départementaux) qui n'ont aucune correspondance lexicale dans le répertoire SIRENE national. Ce ne sont pas des défaillances de ranking, mais des cas relevant par nature d'une résolution contextuelle ou d'un enrichissement SIREN/Places.

---

## 4. 🛡️ Efficacité Validée des Garde-fous de Rétention

Sur l'échantillon des cas éliminés du Top-50 testés avec le module [`src/xgb_matcher/retention.py`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/src/xgb_matcher/retention.py) (Guards 1, 2 et 3) :
- **Cas emblématique repêché** : **`PRESTIGE IMEX AUTOS`** (QID `86565`, Antibes).
  - En baseline : Éliminé au **rang 73** (inondé par des dizaines de SCI / holdings hébergées au 55 avenue de Cannes).
  - Avec les Garde-fous : **Propulsé au rang 49 (dans le Top-50)** grâce à l'ancrage rareté IDF du token `IMEX` ($IDF = 9.80$) et l'atténuation de densité des colocataires passifs.
- **Budget $K=50$ strict garanti** : Aucun débordement budgétaire.
