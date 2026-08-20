# Master Rules for Sireto.dev Development

Ces règles directrices gouvernent l'ensemble des processus de développement, d'analyse, d'expérimentation et de déploiement sur le projet **Sireto.dev**.

---

## 1. Analyse factuelle et basée sur les résultats objectifs
- **Zéro estimation sans fondement démontré** : Toute conclusion, métrique ou hypothèse doit s'appuyer sur des données empiriques mesurables et vérifiables.
- **Référencement obligatoire des sources** : Toujours citer les sources précises (jeux de données, partitions SIRENE, logs d'évaluation, commits GitHub, sorties de scripts).

---

## 2. Aucune modification non autorisée
- **Validation préalable requise** : Toute modification majeure du code, de l'architecture, du pipeline ou des hyperparamètres doit obligatoirement faire l'objet d'une proposition motivée, accompagnée de preuves tangibles (diffs, impacts attendus, justifications techniques).
- **Attente d'autorisation** : L'implémentation effective ne débute qu'après validation explicite de l'utilisateur.

---

## 3. Exécution locale des tests majeurs & conservation des crédits
- **Tests lourds et full runs en local** : Afin de préserver les crédits et ressources, les benchmarks volumineux, les générations massives de samples et les entraînements complets sont exécutés par l'utilisateur sur sa machine locale.
- **Scripts `.bat` / scripts de lancement** : Fournir les fichiers `.bat` ou commandes prêtes à l'emploi avec toutes les options nécessaires pour permettre une exécution autonome par l'utilisateur.

---

## 4. Système de documentation complet et rigoureux
Toute modification doit être adossée à une documentation structurée couvrant :
1. **Implementation Plans** : Plans de mise en œuvre préalables avec analyse d'impact.
2. **Walkthroughs** : Guides et revues post-implémentation détaillant le travail réalisé.
3. **Matching Rules & Strategy** : Règles de matching déterministes, fallbacks, expansions et stratégies de ranking.
4. **Weights Training** : Paramètres d'apprentissage, loss, hyperparamètres XGBoost et artefacts de modèles.
5. **Architectural Structure & Guides** : Schémas et documentation d'architecture des pipelines (V7, V8b, etc.).
6. **Analysis & Performance** : Analyse comparative des métriques, latence, couverture, profiling.
7. **History & Timelines** : Tenue à jour du journal `handover.md` avec citation systématique des commits GitHub associés.
8. **Proposals & Notes** : Notes d'exploration, propositions d'améliorations et arbitrages.
9. **Evaluation & Reverification** : Protocoles d'évaluation, détection de régressions et vérification croisée.
10. **Code Evaluation** : Revues de code, propreté, gestion des dépendances et non-régression.

---

## 5. Intégrité et clarté du référentiel GitHub (Sireto.dev)
- Tous les commits, modifications de code et versions d'artefacts doivent être organisés proprement et traçables dans le dépôt GitHub `Sireto.dev`.
- Chaque évolution métier ou de code doit être référencée dans `handover.md` avec son hash de commit GitHub.

---

## 6. Métriques standardisées après chaque test run
Dès transmission des résultats d'un test run, le rapport doit obligatoirement restituer les marqueurs suivants :
- **Retrieval elimination** : Taux d'élimination / couverture de pool (GT trouvé dans le pool top-k).
- **Taux de répartition du routing** :
  - `Auto Match Rate` (%)
  - `Review Rate` (%)
  - `No Match Rate` (%)
- **Automatch Precision** : Précision réelle sur la population routée en AUTO (objectif zéro faux positif).
- **Précision SIREN & SIRET** :
  - SIREN Precision (identification de l'entité juridique)
  - SIRET Precision (identification de l'établissement exact)
