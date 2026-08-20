# Audit Qualite — CRM Increment 20260817

**Date d'audit** : 20 Aout 2026
**Branche** : feature/geo-resolution-and-crm-expansion
**Fichier source** : `crm_ok_gt_increment_unique_unambiguous_20260817.csv`

---

## 1. Vue d'ensemble du fichier source

| Indicateur | Valeur |
|---|---|
| Total lignes | 20 209 |
| Deja presents dans le dataset exact | 1 449 (7.2%) |
| Deja presents (non exact) | 18 760 (92.8%) |

## 2. Segmentation par relation avec le dataset existant

| Relation | Volume |
|---|---|
| UNSEEN_SIREN_NEEDS_ASSIGNMENT | **15 516** (coeur de valeur) |
| QUARANTINE_EXISTING_NONTRAIN_COMPONENT | 3 221 |
| EXISTING_TRAIN_COMPONENT | 1 472 |

## 3. Criteres qualite sur les 15 516 cas UNSEEN

| Critere | Valeur | Verdict |
|---|---|---|
| crm_name vide | 0 | PASS |
| crm_adresse vide | 0 | PASS |
| crm_cp vide | 0 | PASS |
| Format SIRET valide | 0 erreur | PASS |
| ambiguous_exact_truth_for_crm_surface = True | 0 | PASS |

### Distribution loc_match_type

| loc_match_type | Volume | Pct |
|---|---|---|
| INSEE_AND_CP | 14 607 | 94.1% |
| INSEE_ONLY | 906 | 5.8% |
| CP_FALLBACK_INSEE_MISSING | 3 | 0.02% |

Les 3 cas CP_FALLBACK_INSEE_MISSING sont marginaux et seront traites correctement
par la nouvelle methode load_with_geo_resolution() du Chantier 1.

### Distribution sirene_etat

| Etat | Volume | Pct |
|---|---|---|
| Actif (A) | 12 505 | 80.6% |
| Ferme (F) | 3 011 | 19.4% |

Ratio Actif/Ferme similaire au dataset existant (81.16% / 18.84%). Pas de biais.

### Distribution source_product_category

| Categorie | Volume |
|---|---|
| Fibre Essentiel | 10 547 |
| Revente Fibre Essentiel AI | 2 620 |
| Revente ADSL | 1 313 |
| ADSL | 744 |
| Fibre Confort | 137 |
| SDSL | 108 |
| Fibre Premium | 46 |

## 4. Chevauchement avec le dataset existant

| Indicateur | Valeur |
|---|---|
| SIRENs dans le dataset existant | 14 198 |
| SIRENs dans les cas UNSEEN | 11 361 |
| SIRENs en commun | **0** |
| SIRENs 100% nouveaux | 11 361 (100%) |

**Conclusion : Zero leakage possible. Tous les SIRENs de l'increment sont nouveaux.**
Le split Train/Dev/Test par groupe SIREN peut etre refait sans risque de contamination.

## 5. Verification de securite (deduplication)

- Deduplication sur `crm_gt_fingerprint` entre dataset existant et increment :
  **0 doublon detecte**
- Deduplication intra-increment : **0 doublon**

## 6. Verdict final

**QUALITE SATISFAISANTE — FUSION AUTORISEE**

Le fichier source repond a tous les criteres de qualite. La fusion peut etre
executee avec le script `scripts/merge_crm_increment.py`.

**Resultat attendu post-fusion :**
- `data/crm_ok_gt_merged_v1.csv` : **32 570 requetes CRM**
- Actifs : 13 841 + 12 505 = 26 346
- Fermes : 3 213 + 3 011 = 6 224
