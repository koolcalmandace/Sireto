# Guide d'Utilisation - Sireto (Rapprochement CRM ↔ SIRENE)

Ce guide récapitule l'installation, la configuration et l'exécution du projet **Sireto** dans un environnement Windows sans droits d'administrateur.

---

## 1. Installation de Python (Sans Droits Administrateur)
Si Python n'est pas encore installé sur votre machine, utilisez l'une des solutions suivantes :

* **Option A (Recommandée) : Microsoft Store**
  Recherchez **Python 3.10** ou **Python 3.11** dans le Microsoft Store et cliquez sur *Obtenir*. L'installation s'effectue dans votre session locale sans demander de privilèges.
* **Option B : Miniconda (Option "Just Me")**
  Téléchargez l'installateur Windows de Miniconda et, lors de l'installation, cochez uniquement l'option **Just Me**. Miniconda sera installé dans votre dossier utilisateur.
* **Option C : WinPython (Portable)**
  Téléchargez une version de WinPython, puis décompressez l'archive dans vos documents ou sur votre bureau. Utilisez la console portable intégrée.

---

## 2. Configuration Initiale (Terminal & Dépendances)
Une fois Python disponible, suivez les instructions ci-dessous pour initialiser le projet :

1. Ouvrez votre terminal PowerShell.
2. Copiez-collez et exécutez le bloc de commandes suivant :
   ```powershell
   # Déplacement dans le dossier du projet
   cd "C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto"

   # Création de l'environnement virtuel local 'venv'
   python -m venv venv

   # Mise à jour de pip
   .\venv\Scripts\python.exe -m pip install --upgrade pip

   # Installation des dépendances standard
   .\venv\Scripts\python.exe -m pip install -r requirements.txt

   # Installation des dépendances supplémentaires du projet
   .\venv\Scripts\python.exe -m pip install xgboost pandas pyyaml rapidfuzz requests pyarrow
   ```

---

## 3. Utilisation du Terminal Lanceur Rapide (`Sireto_Terminal.bat`)
Pour accéder rapidement au projet sans avoir à retaper les chemins d'accès :
1. Ouvrez le dossier : `C:\Users\Kabouassi\.gemini\antigravity\scratch\Sireto`.
2. Double-cliquez sur le fichier **`Sireto_Terminal.bat`** (vous pouvez en faire un raccourci sur votre Bureau).
3. Ce script ouvre une console PowerShell propre, pré-positionnée dans le bon dossier, avec un tableau de bord d'aide en couleur.

---

## 4. Commandes Clés d'Exécution
Puisque la politique Windows restreint l'activation standard des scripts, préfixez toujours vos scripts par le chemin de l'exécutable Python local : `.\venv\Scripts\python.exe`.

* **Lancer le Pipeline de Test V6 (Rapprochement CRM)** :
  ```powershell
  .\venv\Scripts\python.exe scripts/run_pipe_v6.py --crm-path data/crm_ok_gt.csv
  ```
* **Lancer le Routage XGBoost (AUTO / REVIEW)** :
  ```powershell
  .\venv\Scripts\python.exe scripts/route_xgb_results.py --help
  ```
* **Lancer les Tests Unitaires** :
  ```powershell
  .\venv\Scripts\python.exe -m unittest discover -s tests
  ```

---

## 5. Accès aux Données et Fichiers Clés
* **Données d'entrée** : Le fichier CRM principal à traiter est stocké dans [data/crm_ok_gt.csv](file:///C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/data/crm_ok_gt.csv).
* **Configuration** : Le fichier [config.yaml](file:///C:/Users/Kabouassi/.gemini/antigravity/scratch/Sireto/config.yaml) permet de configurer les clés API, les seuils de confiance et les modèles d'exécution.
* **Clés API & Variables d'environnement** : Copiez le fichier `.env.example` sous le nom `.env.local` à la racine pour y insérer vos clés d'API sécurisées (Google CSE, Brave, Serper, etc.).
* **Fichiers de Code Source** :
  * Les modules et la logique de rapprochement se trouvent dans le répertoire `src/`.
  * Les scripts de calculs et d'automatisation sont stockés dans `scripts/`.
