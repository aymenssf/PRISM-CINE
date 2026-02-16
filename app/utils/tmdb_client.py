"""
Client TMDB pour récupérer les posters de films.

Ce module fournit une interface simple pour interroger l'API TMDB (The Movie Database)
et récupérer les URLs des posters de films en haute qualité.

Fonctionnalités :
- Récupération d'URL de poster par TMDB ID
- Fallback automatique vers placeholders si API fail
- Logging des erreurs pour debug
- Support pour batch fetching (all posters)

Usage :
    from app.utils.tmdb_client import tmdb_client

    posters = tmdb_client.get_all_posters(MOVIE_CATALOG)
    # {'movie_1': 'https://image.tmdb.org/t/p/w500/...',  ...}
"""
import requests
import os
import logging
from typing import Optional, Dict

# Import depuis config (même niveau app/)
try:
    from config import (
        TMDB_API_KEY,
        TMDB_BASE_URL,
        TMDB_IMAGE_BASE_URL,
        POSTER_CACHE_DIR
    )
except ImportError:
    # Fallback si config non trouvé
    TMDB_API_KEY = os.getenv("TMDB_API_KEY", "543ffe4922b05675289b199cc14b9335")
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    POSTER_CACHE_DIR = "app/static/cache/posters"

logger = logging.getLogger(__name__)


class TMDBClient:
    """
    Client pour TMDB API.

    Gère la récupération des posters avec fallback automatique si :
    - Clé API manquante
    - Film introuvable
    - Erreur réseau
    - Pas de poster disponible
    """

    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.image_base_url = TMDB_IMAGE_BASE_URL
        self._poster_cache: Dict[int, str] = {}  # Cache en mémoire {tmdb_id: url}

    def get_poster_url(self, tmdb_id: int, fallback: Optional[str] = None) -> str:
        """
        Récupère l'URL du poster pour un film TMDB.

        Args:
            tmdb_id: ID TMDB du film (ex: 603 pour The Matrix)
            fallback: URL de fallback personnalisée (optionnel)

        Returns:
            URL complète du poster TMDB ou placeholder si échec

        Examples:
            >>> client.get_poster_url(603)
            'https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg'

            >>> client.get_poster_url(999999999)  # Film inexistant
            'https://placehold.co/600x900/1a1a1a/666666?text=No+Poster'
        """
        # Cache hit
        if tmdb_id in self._poster_cache:
            return self._poster_cache[tmdb_id]

        # Vérifier clé API
        if not self.api_key or self.api_key == "VOTRE_CLE_API_ICI":
            logger.warning(
                f"TMDB_API_KEY non configurée, utilisation du fallback pour film {tmdb_id}"
            )
            return fallback or f"https://placehold.co/600x900/1a1a1a/666666?text=Movie+{tmdb_id}"

        try:
            # Appel API TMDB /movie/{tmdb_id}
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {"api_key": self.api_key}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            poster_path = data.get("poster_path")

            if poster_path:
                full_url = f"{self.image_base_url}{poster_path}"
                self._poster_cache[tmdb_id] = full_url
                logger.info(f"Poster récupéré pour TMDB {tmdb_id}: {full_url}")
                return full_url
            else:
                logger.warning(f"✗ Pas de poster pour TMDB {tmdb_id}")
                return fallback or f"https://placehold.co/600x900/1a1a1a/666666?text=No+Poster"

        except requests.exceptions.Timeout:
            logger.error(f"✗ Timeout TMDB API pour film {tmdb_id}")
            return fallback or f"https://placehold.co/600x900/1a1a1a/666666?text=Timeout"

        except requests.exceptions.HTTPError as e:
            logger.error(f"✗ HTTP Error TMDB API pour film {tmdb_id}: {e}")
            return fallback or f"https://placehold.co/600x900/1a1a1a/666666?text=HTTP+Error"

        except Exception as e:
            logger.error(f"✗ Erreur TMDB API pour film {tmdb_id}: {e}")
            return fallback or f"https://placehold.co/600x900/1a1a1a/666666?text=Error"

    def get_all_posters(self, movie_catalog: Dict) -> Dict[str, str]:
        """
        Récupère tous les posters pour le catalogue de films.

        Args:
            movie_catalog: Dict {movie_id: {title, genre, tmdb_id}}

        Returns:
            Dict {movie_id: poster_url}

        Examples:
            >>> catalog = {
            ...     "movie_1": {"title": "The Matrix", "genre": "Action", "tmdb_id": 603},
            ...     "movie_2": {"title": "Unknown", "genre": "Drama"}  # Pas de tmdb_id
            ... }
            >>> tmdb_client.get_all_posters(catalog)
            {
                'movie_1': 'https://image.tmdb.org/t/p/w500/...',
                'movie_2': 'https://placehold.co/600x900/1a1a1a/666666?text=Unknown'
            }
        """
        posters = {}
        total = len(movie_catalog)
        logger.info(f"Récupération de {total} posters TMDB...")

        for i, (movie_id, movie_data) in enumerate(movie_catalog.items(), 1):
            tmdb_id = movie_data.get("tmdb_id")
            title = movie_data.get("title", "Unknown")

            if tmdb_id:
                poster_url = self.get_poster_url(tmdb_id)
                posters[movie_id] = poster_url
            else:
                logger.warning(f"✗ Pas de tmdb_id pour {movie_id} ({title})")
                # Fallback avec titre du film
                safe_title = title.replace(' ', '+')
                posters[movie_id] = f"https://placehold.co/600x900/1a1a1a/666666?text={safe_title}"

            # Log progression tous les 10 films
            if i % 10 == 0 or i == total:
                logger.info(f"  Progression: {i}/{total} posters récupérés")

        logger.info(f"✓ {total} posters récupérés avec succès")
        return posters


# ============================================================================
# Instance globale singleton
# ============================================================================
# Créée une seule fois au démarrage de l'application
tmdb_client = TMDBClient()
