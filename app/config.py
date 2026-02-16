"""
Configuration centralisée pour Kinetoscope V2.

Ce module contient toutes les constantes système :
- Configuration TMDB API pour récupération des posters
- Hyperparamètres du système de recommandation (SVD + Bandit)
- Paramètres de cache pour les images
"""
import os

# ============================================================================
# TMDB API Configuration
# ============================================================================
# The Movie Database (TMDB) API permet de récupérer les posters de films.
# Gratuit avec inscription sur https://www.themoviedb.org/
#
# Pour utiliser, exporter la variable d'environnement :
#   export TMDB_API_KEY="votre_cle_ici"
#
# Si non configurée, le système utilisera des placeholders.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "543ffe4922b05675289b199cc14b9335")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"  # Résolution 500px width

# ============================================================================
# System Configuration — Recommandation Engine
# ============================================================================
# Hyperparamètres pour le système hybrid SVD + Epsilon-Greedy Bandit

# Factorisation matricielle SVD
LATENT_FACTORS_K = 10           # Dimensions latentes (k)
LEARNING_RATE = 0.01            # Taux d'apprentissage SGD (alpha)
REGULARIZATION = 0.02           # Régularisation L2 (lambda)

# Multi-Armed Bandit
EPSILON = 0.2                   # Probabilité d'exploration (20%)

# ============================================================================
# Cache Settings (pour images TMDB)
# ============================================================================
POSTER_CACHE_DIR = "app/static/cache/posters"
CACHE_EXPIRY_DAYS = 30
