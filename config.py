"""
Configuration centralisée pour l'application d'analyse de documents
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# CONFIGURATION AZURE / LLM
# ==========================================

MODEL_NAME = os.getenv("MODEL_NAME", "azure/gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "azure/text-embedding-3-small")

# Variables Azure obligatoires
REQUIRED_ENV_VARS = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]

# Configuration du logging LiteLLM
# Valeurs possibles: DEBUG, INFO, WARNING, ERROR
LITELLM_LOG_LEVEL = os.getenv("LITELLM_LOG", "ERROR")  # Par défaut ERROR en production

# ==========================================
# LIMITES & SÉCURITÉ
# ==========================================

# Taille maximale des fichiers (en bytes) - 50 MB par défaut
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 50 * 1024 * 1024))

# Limite de tokens en entrée pour le modèle
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 100000))

# Extensions autorisées
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

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

# Fichier de cache pour les embeddings
CACHE_FILE = "vector_store_cache.pkl"

# Paramètres par défaut pour le chunking
DEFAULT_MAX_CHUNK_TOKENS = 600
DEFAULT_OVERLAP_TOKENS = 120

# Paramètres par défaut pour le retrieval
DEFAULT_TOP_K = 6
DEFAULT_SIM_THRESHOLD = 0.15
DEFAULT_MMR_LAMBDA = 0.7

# Dimension des embeddings (text-embedding-3-small)
EMBEDDING_DIMENSION = 1536
