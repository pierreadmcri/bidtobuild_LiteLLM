# 📖 Guide de Configuration

Ce document décrit tous les paramètres configurables de l'application d'analyse de documents IT.

## 🔧 Variables d'Environnement

Créez un fichier `.env` à la racine du projet avec les variables suivantes :

### Obligatoires

```bash
# Configuration Azure OpenAI
AZURE_API_KEY="votre_cle_api"
AZURE_API_BASE="https://votre-instance.openai.azure.com"
AZURE_API_VERSION="2024-02-01"
```

### Optionnelles

```bash
# Modèles utilisés (par défaut)
MODEL_NAME="azure/gpt-4.1-mini"                     # Modèle de génération
EMBEDDING_MODEL_NAME="azure/text-embedding-3-small" # Modèle d'embeddings

# Limites de sécurité
MAX_FILE_SIZE_BYTES=52428800                        # Taille max fichier (50 MB)
MAX_INPUT_TOKENS=100000                             # Limite tokens en entrée

# Logging
LITELLM_LOG="ERROR"                                 # Niveau de log : DEBUG, INFO, WARNING, ERROR
```

## ⚙️ Paramètres de Performance

### Retry Logic

Configuré dans `config.py` :

```python
MAX_RETRIES = 3              # Nombre de tentatives en cas d'échec
RETRY_BASE_DELAY = 2         # Délai initial (secondes)
RETRY_MAX_DELAY = 16         # Délai maximum (secondes)
```

**Comportement** : En cas d'échec API, le système retente avec un délai exponentiel :
- Tentative 1 : immédiate
- Tentative 2 : après 2s
- Tentative 3 : après 4s
- Échec final : après 8s

### Rate Limiting

```python
NB_WORKERS = 4               # Nombre de workers parallèles pour embeddings
BATCH_SIZE = 10              # Taille des lots d'embeddings
RATE_LIMIT_DELAY = 0.1       # Délai entre requêtes (secondes)
```

**Recommandations** :
- **Azure Tier Standard** : `NB_WORKERS=4`, `RATE_LIMIT_DELAY=0.1`
- **Azure Tier Premium** : `NB_WORKERS=8`, `RATE_LIMIT_DELAY=0.05`
- **En cas de rate limit** : Réduire `NB_WORKERS` ou augmenter `RATE_LIMIT_DELAY`

## 🎯 Paramètres RAG (rag_analysis.py)

### Chunking

Configurables via l'interface Streamlit (sidebar) :

| Paramètre | Défaut | Plage | Description |
|-----------|--------|-------|-------------|
| **Taille des segments** | 600 tokens | 200-1500 | Taille cible d'un chunk |
| **Overlap** | 120 tokens | 0-400 | Contexte partagé entre chunks |

**Impact** :
- ⬆️ **Taille segments** : Moins de chunks, contexte plus complet, mais moins précis
- ⬇️ **Taille segments** : Plus de chunks, recherche plus fine, mais perte de contexte
- ⬆️ **Overlap** : Meilleure continuité, mais redondance et coût accru
- ⬇️ **Overlap** : Moins de redondance, mais risque de perte d'information

**Recommandations par cas d'usage** :

| Type de documents | Taille segment | Overlap |
|-------------------|----------------|---------|
| **Documents techniques** (specs, code) | 400-500 | 80-100 |
| **Documents financiers** (budgets, contrats) | 600-800 | 120-150 |
| **Documents mixtes** (RPO, PTC) | 500-700 | 100-150 |
| **Documents longs** (> 50 pages) | 800-1000 | 150-200 |

### Retrieval

| Paramètre | Défaut | Plage | Description |
|-----------|--------|-------|-------------|
| **Top-K** | 6 | 3-20 | Nombre de chunks utilisés pour la génération |
| **Seuil similarité** | 0.15 | 0.0-1.0 | Score minimum de pertinence |
| **MMR activé** | ✅ Oui | - | Diversification des résultats |
| **MMR Lambda (λ)** | 0.7 | 0.1-0.9 | Balance pertinence/diversité |

**Impact MMR Lambda** :
- **λ = 0.9** : Favorise la pertinence (résultats similaires)
- **λ = 0.5** : Balance équilibrée
- **λ = 0.1** : Favorise la diversité (résultats variés)

**Recommandations** :

| Objectif | Top-K | Seuil | MMR | Lambda |
|----------|-------|-------|-----|--------|
| **Synthèse globale** | 8-12 | 0.10 | ✅ | 0.5-0.6 |
| **Recherche précise** | 3-5 | 0.25 | ❌ | - |
| **Analyse exhaustive** | 15-20 | 0.05 | ✅ | 0.7 |
| **Documents courts** | 3-6 | 0.20 | ❌ | - |

## 🛡️ Sécurité

### Validation des Chemins

L'application bloque automatiquement l'accès à :
- `/etc/`, `/sys/`, `/proc/`, `/root/` (Linux)
- `C:\Windows\`, `C:\Program Files\` (Windows)
- Fichiers sensibles : `.ssh`, `.aws`, `credentials`

**Configuration** : Modifiez `FORBIDDEN_PATH_PATTERNS` dans `config.py`

### Taille des Fichiers

Par défaut : **50 MB maximum par fichier**

Pour modifier :
```bash
# .env
MAX_FILE_SIZE_BYTES=104857600  # 100 MB
```

### Extensions Autorisées

Par défaut : `.pdf`, `.docx`, `.txt`

Pour modifier dans `config.py` :
```python
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
```

## 💰 Estimation des Coûts

### Modèle par Défaut (gpt-4.1-mini)

| Scénario | Input Tokens | Output Tokens | Coût (USD) |
|----------|--------------|---------------|------------|
| **5 documents** | ~30,000 | ~800 | $0.05 |
| **10 documents** | ~60,000 | ~1,500 | $0.10 |
| **20 documents** | ~120,000 | ~2,500 | $0.20 |

**Embeddings** (text-embedding-3-small) :
- ~50 documents × 5 chunks = 250 embeddings
- Coût : ~$0.0001 (négligeable)

### Optimisation des Coûts

1. **Réduire Top-K** : Moins de chunks = moins de tokens
2. **Augmenter seuil similarité** : Filtre plus strict
3. **Utiliser le cache** : Évite de recalculer les embeddings
4. **Chunk size optimal** : 600-800 tokens (bon compromis)

## 🚀 Modes d'Utilisation Recommandés

### Mode Développement

```bash
# .env
LITELLM_LOG="DEBUG"
MAX_FILE_SIZE_BYTES=10485760  # 10 MB
NB_WORKERS=2
```

### Mode Production

```bash
# .env
LITELLM_LOG="ERROR"
MAX_FILE_SIZE_BYTES=52428800  # 50 MB
NB_WORKERS=4
```

### Mode Haute Performance

```bash
# .env
LITELLM_LOG="WARNING"
NB_WORKERS=8
BATCH_SIZE=15
RATE_LIMIT_DELAY=0.05
```

**Sidebar RAG** :
- Taille segment : 800
- Overlap : 150
- Top-K : 10
- MMR activé : Oui (λ=0.6)

## 🧪 Tests

Lancer les tests unitaires :

```bash
# Installer pytest
pip install pytest pytest-mock

# Lancer tous les tests
pytest test_utils.py -v

# Lancer un test spécifique
pytest test_utils.py::TestValidation::test_validate_file_path_valid -v

# Avec couverture de code
pip install pytest-cov
pytest test_utils.py --cov=utils --cov-report=html
```

## 📝 Logs

### Configuration des Logs

Les logs sont écrits dans la sortie standard avec le format :
```
2026-01-14 10:30:15 - utils - INFO - Chemin validé : /home/user/documents
```

### Niveaux de Log

- **DEBUG** : Tous les détails (développement uniquement)
- **INFO** : Événements normaux (scan, validation, API calls)
- **WARNING** : Avertissements (fichiers ignorés, fallbacks)
- **ERROR** : Erreurs critiques (échecs API, validation)

### Filtrer les Logs

```bash
# Voir uniquement les erreurs
streamlit run app.py 2>&1 | grep ERROR

# Sauvegarder les logs
streamlit run app.py 2>&1 | tee app.log
```

## 🔍 Troubleshooting

### Erreur "Rate Limit Exceeded"

**Solution** : Réduire `NB_WORKERS` dans `config.py` :
```python
NB_WORKERS = 2  # Au lieu de 4
```

### Erreur "Token Limit Exceeded"

**Solution** : Réduire `MAX_INPUT_TOKENS` ou `Top-K` :
```python
MAX_INPUT_TOKENS = 50000  # Au lieu de 100000
```

### Erreur "File Too Large"

**Solution** : Augmenter `MAX_FILE_SIZE_BYTES` dans `.env` :
```bash
MAX_FILE_SIZE_BYTES=104857600  # 100 MB
```

### Chunking trop lent

**Solution** :
1. Activer le cache (déjà activé par défaut)
2. Réduire la taille des segments
3. Réduire le nombre de documents

### Résultats RAG non pertinents

**Solution** :
1. Augmenter le seuil de similarité (0.20 - 0.30)
2. Réduire Top-K (3-5)
3. Désactiver MMR pour favoriser la pertinence
4. Vérifier que les documents contiennent bien l'information recherchée

---

Pour plus d'informations, consultez :
- [README.md](README.md) - Vue d'ensemble du projet
- [utils.py](utils.py) - Fonctions utilitaires
- [config.py](config.py) - Configuration centralisée
