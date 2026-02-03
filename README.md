# 🏗️ AI Document Scanner

Une application **Streamlit** intelligente qui automatise l'analyse et la synthèse de documents de construction (BCO, RPO, PTC, BDC).

L'outil scanne un répertoire local, identifie les fichiers pertinents grâce à des motifs (Regex), sélectionne automatiquement la version la plus récente en cas de doublon, et génère une synthèse structurée via un LLM (OpenAI / GPT-4).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![OpenAI](https://img.shields.io/badge/Backend-OpenAI-orange.svg)

## 🚀 Fonctionnalités

### Version Standard (app.py)
- **📂 Scan Intelligent :** Analyse automatique d'un dossier local.
- **🔍 Filtrage Regex :** Détection automatique des types de documents :
  - **RPO** (Run / Build)
  - **PTC** (Proposition / Technique)
  - **BCO** (Budget / Mandays)
  - **BDC** (Bon de Commande)
- **📅 Gestion des Versions :** En cas de fichiers multiples pour un même type, seule la version la plus récente (date de modification) est conservée.
- **📄 Support Multi-formats :** Lecture native des fichiers `.pdf`, `.docx` et `.txt`.
- **💰 Estimation des Tokens :** Calcul précis du coût en tokens.
- **🧠 Synthèse IA :** Génération d'un résumé financier et technique via OpenAI GPT-4.

### Version RAG (rag_analysis.py)
- **✂️ Smart Chunking :** Découpe intelligente des documents avec overlap pour préserver le contexte.
- **🔍 Recherche Vectorielle :** Embeddings + similarité cosinus pour trouver les passages pertinents.
- **🎯 MMR (Maximal Marginal Relevance) :** Diversification des résultats pour éviter la redondance.
- **💾 Cache Multi-niveaux :** Streamlit + disque pour éviter les recalculs coûteux.
- **⚡ Traitement Parallèle :** Embeddings calculés en parallèle avec `ThreadPoolExecutor`.

### Améliorations de Sécurité & Performance
- **🛡️ Validation des Chemins :** Protection contre path traversal attacks.
- **📏 Limite de Taille :** Fichiers volumineux rejetés automatiquement (configurable).
- **🔄 Retry Logic :** Tentatives automatiques avec exponential backoff en cas d'échec API.
- **⏱️ Rate Limiting :** Gestion intelligente des quotas API.
- **📝 Logging Structuré :** Traçabilité complète avec niveaux configurables.
- **⚠️ Gestion d'Erreurs :** Messages d'erreur spécifiques et informatifs.

## 🛠️ Prérequis technique

- **Python 3.11** (Recommandé)
- Accès à une clé API OpenAI
- Un dossier (au même niveau que le script) contenant les fichiers a scanner 

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
  # Configuration OpenAI
  OPENAI_API_KEY="votre_cle"
  OPENAI_API_BASE="https://llmproxy.ai.orange"

  # (Optionnel) Forcer un modèle spécifique
  MODEL_NAME="openai/gpt-4.1-mini"
  EMBEDDING_MODEL_NAME="openai/text-embedding-3-small"
  ```
- Lancer l'app : **streamlit run app.py**

## 🗂️ Comprendre la structure du code

Le projet est volontairement compact pour faciliter la prise en main par des débutants. Voici les fichiers clés et leur rôle :

- **`app.py`** : application principale Streamlit. Elle contient toute la logique de bout en bout :
  - *Configuration* : chargement des variables d'environnement et du modèle (`MODEL_NAME`).
  - *Fonctions utilitaires* :
    - `read_file_content` lit les fichiers `.pdf`, `.docx` et `.txt`.
    - `scan_directory` parcourt un dossier local et renvoie la liste des fichiers avec leur date et taille.
    - `estimate_tokens` estime le coût en tokens.
  - *Logique métier* (`process_files`) : identifie les documents RPO, PTC, BCO et BDC à l'aide de Regex, sélectionne la version la plus récente et charge uniquement son contenu.
  - *Interface* : construit l'expérience Streamlit (saisie du dossier à analyser, barre de progression, tableau récapitulatif, synthèse IA).
- **`requirements.txt`** : liste des dépendances nécessaires (Streamlit, OpenAI, pandas, pypdf, python-docx, etc.).
- **`README.md`** : ce guide d'utilisation et de compréhension.

### Flux de fonctionnement (simplifié)

1. **Saisie du chemin** : l'utilisateur entre un dossier local dans l'interface Streamlit.
2. **Scan des fichiers** : `scan_directory` récolte les métadonnées des fichiers présents.
3. **Filtrage par type** : `process_files` applique les motifs Regex pour repérer RPO/PTC/BCO/BDC, garde la version la plus récente et lit son contenu.
4. **Estimation de coût** : `estimate_tokens` calcule les tokens pour anticiper le coût LLM.
5. **Synthèse IA** : le texte combiné est envoyé au client OpenAI pour générer la synthèse financière et technique affichée à l'écran.

En cas de besoin, tous les noms de fonctions et sections sont commentés dans `app.py` pour faciliter la navigation.

## 📚 Documentation Complète

- **[CONFIGURATION.md](CONFIGURATION.md)** : Guide détaillé de configuration
  - Variables d'environnement
  - Paramètres RAG (chunking, retrieval)
  - Optimisation des performances
  - Estimation des coûts
  - Troubleshooting

- **[config.py](config.py)** : Configuration centralisée
- **[utils.py](utils.py)** : Fonctions utilitaires avec retry, validation, rate limiting
- **[prompts/](prompts/)** : Prompts système externalisés et modifiables

## 🧪 Tests

Des tests unitaires sont disponibles pour valider les fonctions critiques :

```bash
# Installer les dépendances de test
pip install pytest pytest-mock pytest-cov

# Lancer les tests
pytest test_utils.py -v

# Avec couverture de code
pytest test_utils.py --cov=utils --cov-report=html
```

## 🔒 Sécurité

Le projet implémente plusieurs mesures de sécurité :

1. **Validation des chemins** : Protection contre path traversal
2. **Limite de taille** : Fichiers trop volumineux rejetés (50 MB par défaut)
3. **Rate limiting** : Prévention du dépassement de quotas API
4. **Logs sécurisés** : Pas d'exposition des clés API ou données sensibles
5. **Gestion d'erreurs robuste** : Messages informatifs sans révéler d'informations système

## ⚡ Performances

### Mode Standard (app.py)
- Traitement séquentiel
- Idéal pour 5-10 documents
- Temps : ~30-60 secondes

### Mode RAG (rag_analysis.py)
- Traitement parallèle des embeddings (4 workers)
- Cache disque pour réutilisation
- Idéal pour analyses répétées
- Temps initial : ~60-90 secondes
- Temps avec cache : ~5-10 secondes

### Optimisations Recommandées

Pour documents volumineux (> 20 fichiers) :
```python
# config.py
NB_WORKERS = 6  # Augmenter les workers
BATCH_SIZE = 15  # Lots plus grands
```

Pour réseau instable :
```python
# config.py
MAX_RETRIES = 5  # Plus de tentatives
RETRY_MAX_DELAY = 32  # Délai max plus long
```
