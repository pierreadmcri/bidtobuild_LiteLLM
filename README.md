# 🤖 AI Document Scanner

Une application **Streamlit** intelligente qui automatise l'analyse et la synthèse de documents de build (BCO, RPO, PTC, BDC).

L'outil scanne un répertoire local, identifie les fichiers pertinents grâce à des motifs (Regex), sélectionne automatiquement la version la plus récente en cas de doublon, et génère une synthèse structurée via un LLM (OpenAI / GPT-4).

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![OpenAI](https://img.shields.io/badge/Backend-OpenAI-orange.svg)

## 🚀 Fonctionnalités

- **📂 Scan Intelligent :** Analyse automatique d'un dossier local.
- **🔍 Filtrage Regex :** Détection automatique des types de documents (RPO, PTC, BCO, BDC).
- **✂️ Smart Chunking :** Découpe intelligente des documents avec overlap pour préserver le contexte.
- **🔍 Recherche Vectorielle RAG :** Embeddings + similarité cosinus pour trouver les passages pertinents.
- **🎯 MMR (Maximal Marginal Relevance) :** Diversification des résultats pour éviter la redondance.
- **💾 Cache Multi-niveaux :** Streamlit + disque pour éviter les recalculs coûteux.
- **⚡ Traitement Parallèle :** Embeddings calculés en parallèle avec `ThreadPoolExecutor`.
- **📄 Support Multi-formats :** Lecture native des fichiers `.pdf`, `.docx`, `.txt`, `.xlsx`, `.xlsm`.
- **🧠 Synthèse IA :** Génération d'analyses contextuelles via OpenAI GPT-4.

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
- **Structure de dossiers requise :**
  - 📁 **Documents/** : Dossier principal avec un sous-dossier par client (important)
  - 📁 **Cache/** : Stockage de la base vectorielle
  - 📁 **Prompts/** : Contient le fichier `rag_system_prompt.txt` (prompt de rédaction pour l'output) 

## 📦 Installation

1. **Se positionner via terminal a l'endroit ou sera créé le Projet**
2. **Cloner le projet**
   ```bash
   git clone [https://github.com/votre-user/votre-repo.git](https://github.com/votre-user/votre-repo.git)
   cd votre-repo
   ```

## 🤖 Lancer l'app
- Installer **Python 3.11**
- Se positionner dans le repertoire via Terminal
- Créer un environnement virtuel :
  - **Windows** : `py -3.11 -m venv .venv`
  - **Mac/Linux** : `python3.11 -m venv .venv`
- Activer l'environnement virtuel Window : **.venv\Scripts\Activate.ps1** Mac : **source .venv/bin/activate**
- Installer les lib **pip install -r requirements.txt**
- Créer un fichier **`.env`** à la racine avec au minimum votre clé API et, si besoin, le modèle voulu :
  ```bash
  # Configuration OpenAI
  OPENAI_API_KEY="votre_cle"
  OPENAI_API_BASE="https://api.votre-proxy-llm.com"

  # (Optionnel) Forcer un modèle spécifique
  MODEL_NAME="openai/gpt-4.1-mini"
  EMBEDDING_MODEL_NAME="openai/text-embedding-3-small"
  ```
- Lancer l'app : **streamlit run rag_analysis.py**

## 🗂️ Structure du code

Le projet est organisé de manière modulaire pour faciliter la maintenance :

- **`rag_analysis.py`** : Application principale Streamlit avec RAG (Retrieval-Augmented Generation)
  - Interface utilisateur interactive
  - Traitement intelligent des documents avec chunking
  - Recherche vectorielle et génération de réponses contextuelles

- **`config.py`** : Configuration centralisée
  - Paramètres API OpenAI
  - Limites de sécurité
  - Configuration RAG (chunking, retrieval, etc.)

- **`utils.py`** : Fonctions utilitaires réutilisables
  - Wrappers API avec retry et rate limiting
  - Validation et sécurité
  - Extraction de texte (PDF, DOCX, Excel, OCR)

- **`prompts/`** : Prompts système externalisés et modifiables

- **`requirements.txt`** : Dépendances Python (Streamlit, OpenAI, pandas, etc.)

### Flux de fonctionnement RAG

1. **Upload/Scan** : L'utilisateur sélectionne un dossier de documents
2. **Chunking** : Les documents sont découpés en segments intelligents
3. **Embeddings** : Vectorisation des segments (cache disque pour performance)
4. **Prompts** : Le prompt de rédaction est lu
5. **Retrieval** : Recherche des segments les plus pertinents par similarité cosinus
6. **Generation** : Le LLM génère une réponse basée sur les segments récupérés

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

## 🔒 Sécurité

Le projet implémente plusieurs mesures de sécurité :

1. **Validation des chemins** : Protection contre path traversal
2. **Limite de taille** : Fichiers trop volumineux rejetés (50 MB par défaut)
3. **Rate limiting** : Prévention du dépassement de quotas API
4. **Logs sécurisés** : Pas d'exposition des clés API ou données sensibles
5. **Gestion d'erreurs robuste** : Messages informatifs sans révéler d'informations système

## ⚡ Performances

- **Traitement parallèle** : Embeddings calculés avec 4 workers simultanés
- **Cache intelligent** : Réutilisation des embeddings pour les analyses répétées
- **Scalabilité** : Idéal pour corpus de 20+ documents
- **Temps initial** : ~60-90 secondes (création des embeddings)
- **Temps avec cache** : ~5-10 secondes (réutilisation)

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
