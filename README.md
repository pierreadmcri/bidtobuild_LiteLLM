# 🏗️ AI Document Scanner

Une application **Streamlit** intelligente qui automatise l'analyse et la synthèse de documents de construction (BCO, RBO, PTC, BDC).

L'outil scanne un répertoire local, identifie les fichiers pertinents grâce à des motifs (Regex), sélectionne automatiquement la version la plus récente en cas de doublon, et génère une synthèse structurée via un LLM (Azure OpenAI / GPT-4) grâce à **LiteLLM**.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![LiteLLM](https://img.shields.io/badge/Backend-LiteLLM-orange.svg)

## 🚀 Fonctionnalités

- **📂 Scan Intelligent :** Analyse automatique d'un dossier local.
- **🔍 Filtrage Regex :** Détection automatique des types de documents :
  - **RBO** (Run / Build)
  - **PTC** (Proposition / Technique)
  - **BCO** (Budget / Mandays)
  - **BDC** (Bon de Commande)
- **📅 Gestion des Versions :** En cas de fichiers multiples pour un même type, seule la version la plus récente (date de modification) est conservée.
- **📄 Support Multi-formats :** Lecture native des fichiers `.pdf`, `.docx` et `.txt`.
- **💰 Estimation des Tokens :** Calcul du coût en tokens avant envoi au LLM.
- **🧠 Synthèse IA :** Génération d'un résumé financier et technique via Azure OpenAI (ou tout autre modèle supporté par LiteLLM).

## 🛠️ Prérequis technique

- **Python 3.11** (Recommandé)
- Accès à une clé API (Azure OpenAI, OpenAI, etc.)

## 📦 Installation

1. **Cloner le projet**
   ```bash
   git clone [https://github.com/votre-user/votre-repo.git](https://github.com/votre-user/votre-repo.git)
   cd votre-repo
   ```

## 🤖 Lancer l'app
- Installer **Python 3.11**
- Se positionner dans le repertoire via Terminal
- Créer un environnement virtuel ex : **python3.11 -m venv .venv**
- Activer l'environnement virtuel Window : **.venv\Scripts\Activate.ps1** Mac : **source .venv/bin/activate**
- Installer les lib **pip install -r requirements.txt**
- Créer un fichier **`.env`** à la racine avec au minimum votre clé API et, si besoin, le modèle voulu :
  ```bash
  # Exemple Azure OpenAI
  AZURE_API_KEY="votre_cle"
  AZURE_API_BASE="https://votre-instance.openai.azure.com"
  AZURE_API_VERSION="2024-02-01"

  # (Optionnel) Forcer un modèle spécifique
  MODEL_NAME="azure/gpt-4.1-mini"
  ```
- Lancer l'app : **streamlit run app.py**

## 🗂️ Comprendre la structure du code

Le projet est volontairement compact pour faciliter la prise en main par des débutants. Voici les fichiers clés et leur rôle :

- **`app.py`** : application principale Streamlit. Elle contient toute la logique de bout en bout :
  - *Configuration* : chargement des variables d'environnement et du modèle (`MODEL_NAME`).
  - *Fonctions utilitaires* :
    - `read_file_content` lit les fichiers `.pdf`, `.docx` et `.txt`.
    - `scan_directory` parcourt un dossier local et renvoie la liste des fichiers avec leur date et taille.
    - `estimate_tokens` estime le coût en tokens via `litellm.token_counter`.
  - *Logique métier* (`process_files`) : identifie les documents RBO, PTC, BCO et BDC à l'aide de Regex, sélectionne la version la plus récente et charge uniquement son contenu.
  - *Interface* : construit l'expérience Streamlit (saisie du dossier à analyser, barre de progression, tableau récapitulatif, synthèse IA).
- **`requirements.txt`** : liste des dépendances nécessaires (Streamlit, LiteLLM, pandas, pypdf, python-docx, etc.).
- **`README.md`** : ce guide d'utilisation et de compréhension.

### Flux de fonctionnement (simplifié)

1. **Saisie du chemin** : l'utilisateur entre un dossier local dans l'interface Streamlit.
2. **Scan des fichiers** : `scan_directory` récolte les métadonnées des fichiers présents.
3. **Filtrage par type** : `process_files` applique les motifs Regex pour repérer RBO/PTC/BCO/BDC, garde la version la plus récente et lit son contenu.
4. **Estimation de coût** : `estimate_tokens` calcule les tokens pour anticiper le coût LLM.
5. **Synthèse IA** : le texte combiné est envoyé à `litellm.completion` pour générer la synthèse financière et technique affichée à l'écran.

En cas de besoin, tous les noms de fonctions et sections sont commentés dans `app.py` pour faciliter la navigation.
