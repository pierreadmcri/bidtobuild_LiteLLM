"""
Tests unitaires pour les fonctions utilitaires
"""
import pytest
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from utils import (
    validate_file_path,
    validate_file_size,
    ValidationError,
    FileTooLargeError,
    retry_with_exponential_backoff,
    RateLimiter,
    estimate_tokens,
    load_prompt
)
import config


class TestValidation:
    """Tests pour les fonctions de validation"""

    def test_validate_file_path_valid(self, tmp_path):
        """Test validation d'un chemin valide"""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        result = validate_file_path(str(test_dir))
        assert result.exists()
        assert result.is_dir()

    def test_validate_file_path_nonexistent(self):
        """Test validation d'un chemin inexistant"""
        with pytest.raises(ValidationError, match="n'existe pas"):
            validate_file_path("/path/that/does/not/exist")

    def test_validate_file_path_forbidden(self):
        """Test validation avec chemin interdit"""
        with pytest.raises(ValidationError, match="Accès interdit"):
            validate_file_path("/etc/passwd")

    def test_validate_file_size_ok(self, tmp_path):
        """Test validation taille fichier OK"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("a" * 1000)  # 1 KB
        # Ne devrait pas lever d'exception
        validate_file_size(test_file)

    def test_validate_file_size_too_large(self, tmp_path):
        """Test validation fichier trop volumineux"""
        test_file = tmp_path / "large.txt"
        # Simuler un fichier énorme
        large_size = config.MAX_FILE_SIZE_BYTES + 1000

        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value = Mock(st_size=large_size)
            with pytest.raises(FileTooLargeError):
                validate_file_size(test_file)


class TestRetryLogic:
    """Tests pour le retry avec exponential backoff"""

    def test_retry_success_first_attempt(self):
        """Test fonction qui réussit du premier coup"""
        mock_func = Mock(return_value="success")
        decorated = retry_with_exponential_backoff(max_retries=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_after_failures(self):
        """Test fonction qui réussit après quelques échecs"""
        mock_func = Mock(side_effect=[Exception("error1"), Exception("error2"), "success"])
        decorated = retry_with_exponential_backoff(max_retries=3, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_all_attempts_fail(self):
        """Test fonction qui échoue tous les essais"""
        mock_func = Mock(side_effect=Exception("persistent error"))
        decorated = retry_with_exponential_backoff(max_retries=2, base_delay=0.01)(mock_func)

        with pytest.raises(Exception, match="persistent error"):
            decorated()

        assert mock_func.call_count == 2

    def test_retry_exponential_backoff_timing(self):
        """Test que le délai augmente exponentiellement"""
        call_times = []

        def failing_func():
            call_times.append(time.time())
            raise Exception("fail")

        decorated = retry_with_exponential_backoff(
            max_retries=3,
            base_delay=0.1,
            max_delay=0.5
        )(failing_func)

        with pytest.raises(Exception):
            decorated()

        # Vérifier que les délais augmentent
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # Au moins le délai de base


class TestRateLimiter:
    """Tests pour le rate limiter"""

    def test_rate_limiter_delays_calls(self):
        """Test que le rate limiter ajoute bien un délai"""
        limiter = RateLimiter(delay=0.1)

        start = time.time()
        limiter.wait()
        limiter.wait()
        elapsed = time.time() - start

        # Le deuxième appel devrait avoir attendu au moins 0.1s
        assert elapsed >= 0.1

    def test_rate_limiter_no_delay_first_call(self):
        """Test que le premier appel n'a pas de délai"""
        limiter = RateLimiter(delay=0.1)

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        # Premier appel devrait être quasi instantané
        assert elapsed < 0.05


class TestTokenEstimation:
    """Tests pour l'estimation de tokens"""

    def test_estimate_tokens_basic(self):
        """Test estimation basique de tokens"""
        text = "Hello world, this is a test."
        tokens = estimate_tokens(text)

        # Devrait retourner un nombre positif
        assert tokens > 0
        # Approximativement 6-8 tokens
        assert 5 < tokens < 15

    def test_estimate_tokens_empty(self):
        """Test estimation sur texte vide"""
        tokens = estimate_tokens("")
        assert tokens >= 0

    def test_estimate_tokens_long_text(self):
        """Test estimation sur texte long"""
        text = "word " * 1000
        tokens = estimate_tokens(text)
        # Devrait être proche de 1000 tokens
        assert 800 < tokens < 1200


class TestPromptLoading:
    """Tests pour le chargement de prompts"""

    def test_load_prompt_success(self, tmp_path):
        """Test chargement réussi d'un prompt"""
        # Créer un faux fichier de prompt
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        prompt_file = prompt_dir / "test_prompt.txt"
        prompt_file.write_text("This is a test prompt.")

        with patch('utils.Path') as mock_path:
            mock_path.return_value.parent = tmp_path
            prompt = load_prompt("test_prompt.txt")

        assert "test prompt" in prompt.lower()

    def test_load_prompt_missing_file(self):
        """Test chargement d'un fichier inexistant"""
        with pytest.raises(Exception):
            load_prompt("nonexistent_prompt.txt")


# Tests pour les fonctions MMR (de rag_analysis.py)
class TestMMR:
    """Tests pour l'algorithme MMR"""

    def test_mmr_basic(self):
        """Test MMR basique avec quelques vecteurs"""
        # Import local pour éviter les dépendances circulaires
        from rag_analysis import mmr

        # Créer des embeddings factices
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similaire au premier
            [0.0, 1.0, 0.0],  # Différent
            [0.0, 0.0, 1.0],  # Très différent
        ])

        query_emb = np.array([1.0, 0.0, 0.0])

        # Demander 2 résultats avec lambda = 0.7 (favorise pertinence)
        results = mmr(embeddings, query_emb, k=2, lambda_mult=0.7)

        assert len(results) == 2
        # Le premier devrait être l'indice 0 (le plus similaire)
        assert results[0] == 0

    def test_mmr_empty_embeddings(self):
        """Test MMR avec embeddings vides"""
        from rag_analysis import mmr

        embeddings = np.array([]).reshape(0, 3)
        query_emb = np.array([1.0, 0.0, 0.0])

        results = mmr(embeddings, query_emb, k=5, lambda_mult=0.5)

        assert len(results) == 0

    def test_mmr_k_greater_than_n(self):
        """Test MMR quand k > nombre d'embeddings"""
        from rag_analysis import mmr

        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        query_emb = np.array([1.0, 0.0])

        # Demander plus de résultats qu'il n'y a d'embeddings
        results = mmr(embeddings, query_emb, k=10, lambda_mult=0.5)

        # Ne devrait retourner que 2 résultats
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
