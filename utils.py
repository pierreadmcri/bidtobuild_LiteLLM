"""
Fonctions utilitaires pour la gestion des erreurs, retry, validation et rate limiting
"""
import os
import time
import logging
from pathlib import Path
from functools import wraps
from typing import Callable, Any, List, Union
from litellm import completion, embedding, token_counter
from PIL import Image
import pytesseract
import config

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# VALIDATION & SÉCURITÉ
# ==========================================

class ValidationError(Exception):
    """Exception levée lors d'une erreur de validation"""
    pass

class FileTooLargeError(Exception):
    """Exception levée quand un fichier dépasse la taille maximale"""
    pass

def validate_file_path(path: str) -> Path:
    """
    Valide un chemin de fichier/dossier pour éviter les path traversal attacks

    Args:
        path: Chemin à valider

    Returns:
        Path object résolu et validé

    Raises:
        ValidationError: Si le chemin est invalide ou dangereux
    """
    try:
        # Résoudre le chemin absolu
        resolved_path = Path(path).resolve()

        # Vérifier que le chemin existe
        if not resolved_path.exists():
            raise ValidationError(f"Le chemin n'existe pas : {path}")

        # Vérifier les patterns interdits
        path_str = str(resolved_path)
        for forbidden in config.FORBIDDEN_PATH_PATTERNS:
            if forbidden.lower() in path_str.lower():
                raise ValidationError(f"Accès interdit à ce chemin : {path}")

        logger.info(f"Chemin validé : {resolved_path}")
        return resolved_path

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Erreur de validation du chemin : {str(e)}")

def validate_file_size(filepath: Path) -> None:
    """
    Vérifie que la taille du fichier ne dépasse pas la limite

    Args:
        filepath: Chemin du fichier à vérifier

    Raises:
        FileTooLargeError: Si le fichier est trop volumineux
    """
    try:
        file_size = filepath.stat().st_size
        max_size = config.MAX_FILE_SIZE_BYTES

        if file_size > max_size:
            size_mb = file_size / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            raise FileTooLargeError(
                f"Fichier trop volumineux : {size_mb:.2f} MB (max: {max_mb:.2f} MB)"
            )

    except FileTooLargeError:
        raise
    except Exception as e:
        logger.warning(f"Impossible de vérifier la taille du fichier : {e}")

# ==========================================
# RETRY LOGIC avec EXPONENTIAL BACKOFF
# ==========================================

def retry_with_exponential_backoff(
    max_retries: int = config.MAX_RETRIES,
    base_delay: float = config.RETRY_BASE_DELAY,
    max_delay: float = config.RETRY_MAX_DELAY,
    exceptions: tuple = (Exception,)
):
    """
    Décorateur pour implémenter retry avec exponential backoff

    Args:
        max_retries: Nombre maximum de tentatives
        base_delay: Délai initial en secondes
        max_delay: Délai maximum en secondes
        exceptions: Tuple d'exceptions à intercepter

    Returns:
        Fonction décorée avec retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None

            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    attempt += 1
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} a échoué après {max_retries} tentatives: {e}"
                        )
                        raise

                    # Calcul du délai avec exponential backoff
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

                    logger.warning(
                        f"{func.__name__} tentative {attempt}/{max_retries} échouée. "
                        f"Nouvelle tentative dans {delay}s: {e}"
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator

# ==========================================
# RATE LIMITER
# ==========================================

class RateLimiter:
    """
    Rate limiter simple pour éviter de dépasser les quotas API
    """
    def __init__(self, delay: float = config.RATE_LIMIT_DELAY):
        self.delay = delay
        self.last_call = 0

    def wait(self):
        """Attend le délai nécessaire avant le prochain appel"""
        current_time = time.time()
        elapsed = current_time - self.last_call

        if elapsed < self.delay:
            wait_time = self.delay - elapsed
            time.sleep(wait_time)

        self.last_call = time.time()

# Instance globale du rate limiter
rate_limiter = RateLimiter()

# ==========================================
# WRAPPERS API avec RETRY & RATE LIMITING
# ==========================================

@retry_with_exponential_backoff()
def safe_completion(*args, **kwargs):
    """
    Wrapper sécurisé pour litellm.completion avec retry et rate limiting
    """
    rate_limiter.wait()

    try:
        response = completion(*args, **kwargs)
        logger.info(f"Completion réussie avec le modèle {kwargs.get('model', 'unknown')}")
        return response

    except Exception as e:
        logger.error(f"Erreur lors de l'appel completion: {e}")
        raise

@retry_with_exponential_backoff()
def safe_embedding(texts: List[str], model: str = None, **kwargs):
    """
    Wrapper sécurisé pour litellm.embedding avec retry et rate limiting

    Args:
        texts: Liste de textes à embedder (ou texte unique)
        model: Nom du modèle d'embedding
        **kwargs: Arguments supplémentaires pour l'API

    Returns:
        Liste d'embeddings

    Raises:
        Exception: Si tous les retries échouent
    """
    rate_limiter.wait()

    model = model or config.EMBEDDING_MODEL_NAME

    # Troncature de sécurité basée sur les tokens réels
    safe_texts = []
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        try:
            # Utiliser token_counter pour une estimation précise
            token_count = token_counter(model=model, text=text)
            # Limite API: ~8191 tokens pour text-embedding-3-small
            if token_count > 8000:
                # Troncature approximative (4 chars ≈ 1 token)
                safe_text = text[:32000]
            else:
                safe_text = text
            safe_texts.append(safe_text)
        except Exception:
            # Fallback sur troncature simple
            safe_texts.append(text[:32000])

    try:
        response = embedding(
            model=model,
            input=safe_texts,
            **kwargs
        )

        # Extraction robuste des embeddings
        if hasattr(response, "data"):
            embeddings = [
                d["embedding"] if isinstance(d, dict) else d.embedding
                for d in response.data
            ]
        elif isinstance(response, dict) and "data" in response:
            embeddings = [d["embedding"] for d in response["data"]]
        else:
            raise ValueError("Format de réponse d'embedding inattendu")

        logger.info(f"Embedding réussi pour {len(texts)} texte(s)")
        return embeddings

    except Exception as e:
        logger.error(f"Erreur lors de l'appel embedding: {e}")
        raise

# ==========================================
# EXTRACTION DE TEXTE (OCR)
# ==========================================

def extract_text_from_image(image_path: Union[str, Path], lang: str = None) -> str:
    """
    Extrait le texte d'une image avec Tesseract OCR (gratuit, sans API)

    Args:
        image_path: Chemin vers l'image
        lang: Langue(s) pour l'OCR (défaut: config.OCR_LANGUAGE)
              Format: 'fra' pour français, 'eng' pour anglais, 'fra+eng' pour les deux

    Returns:
        Texte extrait de l'image

    Raises:
        Exception: Si l'OCR échoue
    """
    if not config.ENABLE_IMAGE_OCR:
        logger.info("OCR désactivé")
        return "[Image présente - OCR désactivé]"

    lang = lang or config.OCR_LANGUAGE

    try:
        # Ouvrir l'image
        img = Image.open(image_path)

        # Amélioration de la qualité pour OCR
        # Upscaling si l'image est trop petite
        min_dim = config.MIN_OCR_DIMENSION
        if min(img.size) < min_dim:
            ratio = min_dim / min(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Image agrandie à {new_size} pour améliorer l'OCR")

        # Conversion en niveaux de gris pour améliorer la reconnaissance
        if img.mode != 'L':
            img = img.convert('L')

        # Extraction du texte avec Tesseract
        text = pytesseract.image_to_string(img, lang=lang)

        # Nettoyer le texte extrait
        text = text.strip()

        if text:
            logger.info(f"OCR réussi : {len(text)} caractères extraits de {image_path}")
            return text
        else:
            logger.warning(f"Aucun texte détecté dans l'image : {image_path}")
            return "[Image sans texte détectable]"

    except pytesseract.TesseractNotFoundError:
        error_msg = (
            "Tesseract OCR n'est pas installé. "
            "Installation requise : sudo apt-get install tesseract-ocr tesseract-ocr-fra"
        )
        logger.error(error_msg)
        return f"[Erreur OCR : Tesseract non installé]"

    except Exception as e:
        logger.error(f"Erreur lors de l'OCR de l'image {image_path}: {e}")
        return f"[Erreur OCR : {str(e)}]"

# ==========================================
# GESTION DES TOKENS
# ==========================================

def estimate_tokens(text: str, model: str = None) -> int:
    """
    Estimation précise du nombre de tokens via litellm.token_counter

    Args:
        text: Texte à analyser
        model: Nom du modèle (optionnel)

    Returns:
        Nombre de tokens
    """
    model = model or config.MODEL_NAME

    try:
        return token_counter(model=model, text=text)
    except Exception as e:
        logger.warning(f"Erreur token_counter, utilisation d'une approximation: {e}")
        # Fallback sur approximation simple
        return max(len(text) // 4, 1)

# ==========================================
# CALCUL DES COÛTS
# ==========================================

def calculate_cost(input_tokens: int, output_tokens: int = 0, model: str = None) -> dict:
    """
    Calcule le coût estimé pour un appel API

    Args:
        input_tokens: Nombre de tokens en entrée
        output_tokens: Nombre de tokens en sortie (0 par défaut)
        model: Nom du modèle (optionnel, utilise MODEL_NAME par défaut)

    Returns:
        Dictionnaire avec coûts détaillés:
        {
            "input_cost": float,      # Coût entrée en USD
            "output_cost": float,     # Coût sortie en USD
            "total_cost": float,      # Coût total en USD
            "input_tokens": int,      # Tokens entrée
            "output_tokens": int,     # Tokens sortie
            "total_tokens": int,      # Total tokens
            "model": str              # Modèle utilisé
        }
    """
    model = model or config.MODEL_NAME

    # Récupérer les tarifs du modèle
    pricing = config.PRICING.get(model, config.DEFAULT_PRICING)

    # Calcul des coûts (tarifs par 1000 tokens)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "model": model
    }

def format_cost(cost: float) -> str:
    """
    Formate un coût en USD pour l'affichage

    Args:
        cost: Coût en USD

    Returns:
        Chaîne formatée (ex: "$0.0015" ou "$1.50")
    """
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1.0:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"

# ==========================================
# CHARGEMENT DES PROMPTS
# ==========================================

def load_prompt(filename: str) -> str:
    """
    Charge un prompt depuis le dossier prompts/

    Args:
        filename: Nom du fichier prompt (ex: 'app_system_prompt.txt')

    Returns:
        Contenu du prompt

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    prompt_path = Path(__file__).parent / "prompts" / filename

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        logger.info(f"Prompt chargé : {filename}")
        return prompt
    except Exception as e:
        logger.error(f"Impossible de charger le prompt {filename}: {e}")
        raise
