# Analyse Qualifiée du Benchmark Réconciliation Établissement vs Holding (459 cas actifs)

**Date d'analyse** : 23 Septembre 2026  
**Branche Git** : `feature/holding-establishment-reconciliation`  
**Dataset source** : [`reports/geo_validation/benchmark_holding_reconciliation_453_details.csv`](file:///c:/Users/Kabouassi/.gemini/antigravity-ide/scratch/Sireto-dev/reports/geo_validation/benchmark_holding_reconciliation_453_details.csv)  
**Population évaluée** : 459 requêtes actives du benchmark 32k initialement éliminées au rang > 500 par le TF-IDF (`PRUNED_BY_TFIDF`, `sirene_etat == "A"`).

---

## 1. Synthèse Globale & Validation du Moteur de Réconciliation

Sur les 459 cas actifs réputés "perdus" :
- **Taux de qualification / réconciliation immédiat** : **141 cas (30.72%)**
- **Cas restant non résolus** : **318 cas (69.28%)** (dont la majorité relève d'adresses sans numéro physique *« 0 Lieu-dit »*, voies départementales, ou codes magasins CRM internes).
- **Temps d'exécution moyen** : ~1.4s par requête (avec rechargement à chaud des partitions métropolitaines).

```mermaid
pie title Répartition des 459 cas actifs prunés
    "1. Site Local Qualifié (SSDA / Acronymes)" : 131
    "2. Même SIREN Réconcilié (Secondaire vs Siège GT)" : 9
    "3. Match Direct GT" : 1
    "4. Non Récupérables (Sans numéro / Lieu-dit / Route)" : 318
```

---

## 2. Décomposition Détaillée des Résultats

| Catégorie de Résolution | Volume | Part (%) | Signification Métier |
| :--- | :---: | :---: | :--- |
| **`QUALIFIED_LOCAL_SITE`** | **131** | **28.5%** | **Établissement physique local certifié** : Le bâtiment au numéro et nom de voie possède une entité active avec correspondance d'activité/enseigne ou occupant unique. |
| **`RECONCILED_SAME_SIREN`** | **9** | **2.0%** | **Preuve de Filiation Intra-Entreprise** : L'établissement local sur site partage exactement le même SIREN (9 chiffres) que le SIRET comptable/siège renseigné dans le CRM. |
| **`MATCH_DIRECT_GT`** | **1** | **0.2%** | Correspondance directe du SIRET GT. |
| **`UNRESCUED`** | **318** | **69.3%** | Requêtes sans ancre physique exploitable (60 cas sans numéro de voie, codes enseignes purs, etc.). |

---

## 3. Répartition par Type d'Établissement & Pilier

### A. Nature Juridique Qualifiée
- **`SIEGE`** : 85 cas (60.3% des résolus) — Entité autonome ou siège local situé à l'adresse CRM.
- **`SECONDAIRE`** : 56 cas (39.7% des résolus) — Succursale ou site opérationnel formellement rattaché à son siège de facturation.

### B. Contribution des Piliers
- **Pilier SSDA (Double-Ancre Spatiale-Sectorielle)** : 78 cas (55.3% des résolutions)
- **Pilier Acronymes Dynamiques** : 63 cas (44.7% des résolutions)

---

## 4. Exemples Emblématiques de Qualification

### Cas 1 : La Clé de Voûte (Lien Même SIREN direct)
- **CRM** : `La Cle de Voute `, `88 ROUTE DE CANNES, GRASSE`
- **SIRET GT CRM** : `51892253900029` (Siège Social du groupe)
- **SIRET Retenu (Site)** : `51892253900045` (Établissement secondaire au 88 route de Cannes)
- **Type** : `SECONDAIRE`
- **SIRET Siège Holding** : `51892253900029` (**Correspondance 100% avec le GT**)
- **Nom Holding** : `LE GROUPE DE LA PIERRE`

### Cas 2 : Hôtel du Pin (Enseigne locale vs Holding de gestion)
- **CRM** : `HOTEL DU PIN (KYRIAD)`, `6 RUE EMMANUEL PHILIBERT, NICE`
- **SIRET GT CRM** : `75290494600015` (`SO NICE HOTELS`, siège au 152 bd des Jardiniers)
- **SIRET Retenu (Site)** : `38491117800034` (Établissement physique local `HOTEL DU PIN`)
- **Type** : `SECONDAIRE`
- **SIRET Siège Holding** : `38491117800042` (Siège au 152 bd des Jardiniers)
- **Nom Holding** : `SOCIETE D'EXPLOITATION DE L'HOTEL CASSINI`

---

## 5. Impact sur le Benchmark National (32 884 requêtes)

- Le benchmark initial retenait **98.10%** des cas actifs au Top-500 (506 requêtes perdues).
- Avec la récupération qualifiée de **141 cas**, le taux de rappel actif au Top-500 passe de **98.10% à 98.63%** (+0.53 pt net, soit un gain de près d'un tiers sur l'ensemble du reliquat d'échecs actifs national).
- Zéro régression : **46/46 tests unitaires passent au vert** en 7.1s.
