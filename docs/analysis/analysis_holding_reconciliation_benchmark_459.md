# Analyse Qualifiée du Benchmark Réconciliation Établissement vs Holding (459 cas actifs)

**Date d'analyse** : 23 Septembre 2026  
**Branche Git** : `feature/holding-establishment-reconciliation`  
**Dataset source** : [`reports/geo_validation/benchmark_holding_reconciliation_453_details.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/benchmark_holding_reconciliation_453_details.csv)  
**Population évaluée** : 459 requêtes actives du benchmark 32k initialement éliminées au rang > 500 par le TF-IDF (`PRUNED_BY_TFIDF`, `sirene_etat == "A"`).

---

## 1. Synthèse Globale & Validation du Moteur de Réconciliation (v2 : Noms Civils & Spécificité Multi-Occupants)

Sur les 459 cas actifs réputés "perdus" :
- **Taux de qualification / réconciliation certifié** : **158 cas (34.42%)** (hausse nette de +17 cas par rapport aux 141 initiaux)
- **Cas restant non résolus** : **301 cas (65.58%)** (dont 128 cas masqués RGPD `[ND]`, 75 cas de relocalisation/dérive d'adresse gérés par le fallback Pipe V7 Places, et 61 cas sans numéro physique / lieu-dit).
- **Temps d'exécution moyen** : ~1.8s par requête (avec rechargement à chaud des partitions métropolitaines).

```mermaid
pie title Répartition des 459 cas actifs prunés (v2)
    "1. Site Local Qualifié (SSDA / Acronymes)" : 140
    "2. Même SIREN Réconcilié (Secondaire vs Siège GT)" : 15
    "3. Match Direct GT" : 3
    "4. Non Récupérables Déterministes (RGPD / Dérive Places / Sans numéro)" : 301
```

---

## 2. Décomposition Détaillée des Résultats

| Catégorie de Résolution | Volume Initial | Volume v2 (Actuel) | Évolution | Part (%) | Signification Métier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`QUALIFIED_LOCAL_SITE`** | 131 | **140** | **+9** | **30.5%** | **Établissement physique local certifié** : Bâtiment au numéro et nom de voie avec activité/enseigne concordante ou occupant unique. |
| **`RECONCILED_SAME_SIREN`** | 9 | **15** | **+6 (+67%)** | **3.3%** | **Preuve de Filiation Intra-Entreprise** : L'établissement local sur site partage exactement le même SIREN (9 chiffres) que le SIRET comptable/siège CRM. |
| **`MATCH_DIRECT_GT`** | 1 | **3** | **+2 (+200%)** | **0.7%** | Correspondance directe et absolue du SIRET GT CRM. |
| **Total Qualifié / Réconcilié** | 141 | **158** | **+17** | **34.4%** | **Succès total déterministe** sans aucun dictionnaire statique. |
| **`UNRESCUED`** | 318 | **301** | **-17** | **65.6%** | Cas protégés (RGPD [ND], dérive d'adresse pour Google Places, sans numéro). |

---

## 3. Répartition par Type d'Établissement & Pilier

### A. Nature Juridique Qualifiée
- **`SIEGE`** : 97 cas (61.4% des résolus) — Entité autonome ou siège local situé à l'adresse CRM.
- **`SECONDAIRE`** : 61 cas (38.6% des résolus) — Succursale ou site opérationnel formellement rattaché à son siège de facturation.

### B. Contribution des Piliers
- **Pilier SSDA (Double-Ancre Spatiale-Sectorielle + Spécificité)** : **98 cas** (62.0% des résolutions, +20 cas grâce à l'intégration des personnes physiques et de la spécificité multi-occupants)
- **Pilier Acronymes Dynamiques** : **60 cas** (38.0% des résolutions)

---

## 4. Exemples Emblématiques de Qualification

### Cas 1 : Personne Physique & Match Direct GT (`D'ASTORG ARCHITECTE`)
- **CRM** : `D'ASTORG ARCHITECTE`, `134 RUE DE RIVOLI, PARIS 1ER`
- **SIRET GT CRM** : `34111965900058`
- **SIRET Retenu (Site)** : `34111965900058` (**Match Direct GT 100%**)
- **Type** : `SIEGE`
- **Nom Holding / Civil** : `GUILLAUME D'ASTORG` (extrait de `nom_ul` + `prenom_usuel_ul`)

### Cas 2 : Profession Libérale Médicale (`DR BECQUET`)
- **CRM** : `DR BECQUET`, `10 RUE DU CHATEAU D'EAU`
- **SIRET GT CRM** : `42501378600024` (Ancien site d'exercice)
- **SIRET Retenu (Site)** : `42501378600081` (Cabinet actuel sur site)
- **Résolution** : `RECONCILED_SAME_SIREN` (Même SIREN `425013786`)
- **Nom Civil** : `MURIELLE BECQUET`

### Cas 3 : Cabinet d'Avocat (`Mme DURAND DASTES CLAIRE`)
- **CRM** : `Mme DURAND DASTES CLAIRE`, `15 RUE BUFFON`
- **SIRET GT CRM** : `81059865600058`
- **SIRET Retenu (Site)** : `81059865600066`
- **Résolution** : `RECONCILED_SAME_SIREN` (Même SIREN `810598656`)
- **Nom Civil** : `CLAIRE DURAND DIT DURAND DASTES`

### Cas 4 : Multi-Occupant avec Départage par Spécificité Nom Propre (`RODRIGUEZ GARAGE LAROQUE`)
- **CRM** : `RODRIGUEZ GARAGE LAROQUE`, `63 AVENUE COTE VERMEILLE`
- **Candidats au 63 Avenue** : 2 garages automobiles actifs au même numéro (`RODRIGUEZ CHRISTOPHER / GARAGE LAROQUE` vs `SARL GARAGE COTE VERMEILLE`)
- **Résolution** : La surpondération des noms propres rares (`RODRIGUEZ`, `LAROQUE`) par rapport aux termes génériques (`GARAGE`) permet de départager avec un écart de score $\ge 1.5$ et de qualifier le SIRET exact `89175624900019`.

### Cas 5 : Foyer & Établissement Public Communal (`Foyer Logement Paul Schrive`)
- **CRM** : `Foyer Logement Paul Schrive`, `22 rue Seurat`
- **SIRET GT CRM** : `26590155300098`
- **SIRET Retenu (Site)** : `26590155300064`
- **Résolution** : `RECONCILED_SAME_SIREN` (Même SIREN `265901553`)
- **Nom Holding** : `CENTRE COMMUNAL ACTION SOCIALE DE COUDEKERQUE`

---

## 5. Impact sur le Benchmark National (32 884 requêtes)

- Le benchmark national initial retenait **98.10%** des cas actifs au Top-500 (506 requêtes perdues sur l'ensemble du territoire).
- Avec la récupération certifiée de **158 cas** via le Sidecar réconciliateur, le taux de rappel actif national au Top-500 passe de **98.10% à 98.69%** (+0.59 pt net).
- Cela représente le sauvetage de **31.2% de l'ensemble des cas d'échec actifs métropolitains**.
- Zéro régression : **49/49 tests unitaires passent au vert** en 5.73s sur l'ensemble de la suite Sidecar, NAF et Rescued Features.
