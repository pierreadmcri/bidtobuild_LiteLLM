"""
Configuration centralisée pour l'application d'analyse de documents
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION OPENAI / LLM
# ==========================================

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "openai/text-embedding-3-small")

# URL du proxy/API OpenAI
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://llmproxy.ai.orange")

# Variables OpenAI obligatoires
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "OPENAI_API_BASE"]

# ==========================================
# LIMITES & SÉCURITÉ
# ==========================================

# Taille maximale des fichiers (en bytes) - 50 MB par défaut
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))

# Limite de tokens en entrée pour le modèle
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 100000))

# Extensions autorisées
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".xlsx", ".xlsm"}

# Chemins interdits (sécurité)
FORBIDDEN_PATH_PATTERNS = [
    "/etc/", "/sys/", "/proc/", "/root/",
    "C:\\Windows\\", "C:\\Program Files\\",
    ".ssh", ".aws", "credentials"
]

# ==========================================
# PERFORMANCE & RATE LIMITING
# ==========================================

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # secondes
RETRY_MAX_DELAY = 16  # secondes

# Rate limiting pour embeddings
NB_WORKERS = 4  # Nombre de workers parallèles
BATCH_SIZE = 10  # Taille des lots d'embeddings
RATE_LIMIT_DELAY = 0.1  # Délai entre chaque requête (secondes)

# ==========================================
# RAG CONFIGURATION
# ==========================================

# Dossier pour les fichiers de cache et temporaires
CACHE_DIR = os.path.join(os.getcwd(), "cache")

# Créer le dossier cache s'il n'existe pas
os.makedirs(CACHE_DIR, exist_ok=True)

# Fichier de cache pour les embeddings
CACHE_FILE = os.path.join(CACHE_DIR, "vector_store_cache.pkl")

# Paramètres par défaut pour le chunking
DEFAULT_MAX_CHUNK_TOKENS = 600
DEFAULT_OVERLAP_TOKENS = 120

# Paramètres par défaut pour le retrieval
DEFAULT_TOP_K = 6
DEFAULT_SIM_THRESHOLD = 0.15
DEFAULT_MMR_LAMBDA = 0.7

# Dimension des embeddings (text-embedding-3-small)
EMBEDDING_DIMENSION = 1536

# ==========================================
# CONFIGURATION OCR (IMAGES)
# ==========================================

# Taille maximale des images (en bytes) - 20 MB par défaut
MAX_IMAGE_SIZE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", 20 * 1024 * 1024))

# Activer/désactiver l'extraction de texte des images (OCR)
ENABLE_IMAGE_OCR = os.getenv("ENABLE_IMAGE_OCR", "true").lower() == "true"

# Langue(s) pour l'OCR (Tesseract)
# Format: 'fra' pour français, 'eng' pour anglais, 'fra+eng' pour les deux
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "fra+eng")

# Résolution minimale pour améliorer la qualité OCR (upscaling)
# Les images plus petites seront agrandies pour améliorer la reconnaissance
MIN_OCR_DIMENSION = int(os.getenv("MIN_OCR_DIMENSION", 1000))

# ==========================================
# CONFIGURATION EXCEL
# ==========================================

# Nombre maximum d'onglets à extraire par fichier Excel (défaut: 2)
MAX_EXCEL_SHEETS = int(os.getenv("MAX_EXCEL_SHEETS", 2))

# Stratégie de sélection des onglets : 'exact' (correspondance stricte), 'auto' (recherche partielle), 'first' (premiers onglets)
EXCEL_SHEET_STRATEGY = os.getenv("EXCEL_SHEET_STRATEGY", "exact")

# Noms d'onglets à prioriser (séparés par des virgules)
# Ex: "Build,Run"
EXCEL_PRIORITY_SHEETS = os.getenv("EXCEL_PRIORITY_SHEETS", "Build,Run")

# Nombre maximum de lignes à extraire par onglet (0 = toutes les lignes)
MAX_EXCEL_ROWS = int(os.getenv("MAX_EXCEL_ROWS", 1000))

# ==========================================
# TARIFICATION OPENAI (USD)
# ==========================================

# Tarifs par 1000 tokens (mise à jour: Janvier 2025)
# Source: https://openai.com/api/pricing/

PRICING = {
    # GPT-4o
    "openai/gpt-4o": {
        "input": 0.0025,    # $2.50 / 1M tokens
        "output": 0.01,     # $10.00 / 1M tokens
    },
    "openai/gpt-4o-mini": {
        "input": 0.00015,   # $0.15 / 1M tokens
        "output": 0.0006,   # $0.60 / 1M tokens
    },
    # GPT-4.1 Mini (modèle déployé)
    "openai/gpt-4.1-mini": {
        "input": 0.00015,   # $0.15 / 1M tokens
        "output": 0.0006,   # $0.60 / 1M tokens
    },
    # GPT-4 Turbo
    "openai/gpt-4-turbo": {
        "input": 0.01,      # $10.00 / 1M tokens
        "output": 0.03,     # $30.00 / 1M tokens
    },
    # GPT-4 (anciennes versions)
    "openai/gpt-4": {
        "input": 0.03,      # $30.00 / 1M tokens
        "output": 0.06,     # $60.00 / 1M tokens
    },
    # GPT-3.5 Turbo
    "openai/gpt-3.5-turbo": {
        "input": 0.0005,    # $0.50 / 1M tokens
        "output": 0.0015,   # $1.50 / 1M tokens
    },
    # Embeddings
    "openai/text-embedding-3-small": {
        "input": 0.00002,   # $0.02 / 1M tokens
        "output": 0.0,
    },
    "openai/text-embedding-3-large": {
        "input": 0.00013,   # $0.13 / 1M tokens
        "output": 0.0,
    },
    "openai/text-embedding-ada-002": {
        "input": 0.0001,    # $0.10 / 1M tokens
        "output": 0.0,
    },
}

# Tarif par défaut si modèle non trouvé (GPT-4.1-mini)
DEFAULT_PRICING = {
    "input": 0.00015,
    "output": 0.0006,
}
