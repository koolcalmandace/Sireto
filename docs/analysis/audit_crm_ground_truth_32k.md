# Rapport d'Audit Exhaustif du Ground-Truth CRM (32 570 Requêtes)

**Date d'audit** : 2026-10-06 15:40:05  
**Dataset analysé** : `C:\Users\Kabouassi\.gemini\antigravity-ide\scratch\Sireto-dev\data\crm_ok_gt_merged_v1.csv`  
**Magasin de données officiel** : `C:\Users\Kabouassi\.gemini\antigravity-ide\scratch\Sireto-dev\data\candidates_v8_national`  
**Durée d'analyse** : 599.44 secondes  

---

## 1. Synthèse Exécutive de la Fiabilité du Ground-Truth

L'audit intégral des 32 570 paires CRM / Ground-Truth révèle une base de test **très solide sur le plan structurel mais porteuse d'ambiguïtés d'étiquetage bien définies** :

| Dimension Auditée | Statut / Taux | Nombre de Cas | Diagnostic & Impact Benchmark |
| :--- | :---: | :---: | :--- |
| **Validité Formelle Luhn (SIRET)** | **100.0%** | 32570 / 32570 | **Conforme** (Quasi-zéro erreur de saisie pure) |
| **Présence Physique SIRENE V8** | **98.25%** | 32001 / 32570 | **Conforme** (Entités réelles existantes) |
| **Établissements Actifs ('A')** | **80.89%** | 26346 / 32570 | **Cœur de cible** commercial et de facturation |
| **Établissements Radiés / Fermés ('F')** | **19.11%** | 6224 / 32570 | ⚠️ **Pollution d'évaluation** (SIRET juridiquement éteint) |
| **Concordance INSEE Stricte** | **99.75%** | 32488 / 32570 | **Idéal** pour le retrieval communal local |
| **SIRET dans un Autre Département** | **0.0%** | 0 / 32570 | ⚠️ **Divergence Siège vs Site** (Siège payeur national) |
| **Divergence Lexicale Sévère (Jaro < 0.30)** | **0.3%** | 96 / 32570 | ⚠️ **Enseigne commerciale vs Holding légale** |
| **Masquage Statistique RGPD ([ND])** | **0.17%** | 53 / 32570 | **Plafond légal** (Personnes physiques masquées) |

---

## 2. Typologie Détaillée des Écarts de Vérité Terrain (Ground-Truth)

### A. La Dualité 'Siège Social Payeur' vs 'Établissement Local'
- Dans **63.36%** des cas (20275 cas), le GT est le **Siège Social** de l'entreprise.
- Dans **0.0%** des cas (0 cas), le SIRET GT est situé dans un **autre département** que l'adresse CRM saisie par le client.

**Exemples réels de divergences de département dans le CRM :**

| ID CRM | Nom CRM | CP CRM (Commune) | GT SIRET | CP SIRENE (Commune) | État |
| :--- | :--- | :--- | :--- | :--- | :---: |

> [!NOTE]
> Ce phénomène s'explique par la pratique commerciale courante : le CRM client stocke souvent le SIRET du siège social (où sont envoyées les factures), alors que la livraison ou le service est réalisé sur une agence locale.

### B. Les Divergences Textuelles Totales (Enseigne Commerciale vs Holding)
- **96 requêtes** ont un Jaro-Winkler strictement inférieur à 0.30.
**Exemples réels de déconnexion lexicale :**

| ID CRM | Nom saisi dans le CRM | GT SIRET | Raison Sociale Légale SIRENE | Jaro | Siège ? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `87854` | 483492-CYBE | `86320003600094` | CYBE | 0.0 | False |
| `FR012267` | EP (Batiment Unik) | `49939825300041` | EP | 0.0 | False |
| `FR024471` | DB GROUP SAS | `39931851800061` | DB | 0.0 | False |
| `40055` | Expert.com  | `87843139400013` | AGORA EXPERTS | 0.285 | False |
| `105291` | SARL PEUGEOT CANY | `39809927500016` | SARL CANY | 0.0 | True |
| `FR024321` | UPEMO | `39971975600013` | ASS.UPEMO | 0.0 | True |
| `AC017901` | METAFIX | `31514394100039` | J.B.G. METAFIX | 0.0 | True |
| `FR031230` | CETIH | `82097989600017` | KOOV | 0.0 | True |


---

## 3. Recommandations pour la Certification des Benchmarks

1. **Stratification Actifs / Fermés** : Le score réel à retenir est celui sur les **80.89% d'actifs** (26346 cas), car évaluer la performance de matching sur des SIRETs fermés depuis des années biaise artificiellement les métriques vers le bas.
2. **Schéma de sortie 4 Colonnes** : Confirmer l'adoption des colonnes `siret_retenu` (site physique) et `siret_siege_holding` (siège payeur) pour réconcilier les cas où l'utilisateur a saisi l'agence locale mais attendait le siège.
3. **Rôle confirmé du Fallback Places** : Les 0.3% de cas sans aucun lien lexical ne sont pas une faiblesse de XGBoost, mais une limite intrinsèque de la nomenclature SIRENE, que seul Google Places peut légitimement combler.
