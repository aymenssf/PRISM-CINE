#!/usr/bin/env python3
"""
Script de démonstration automatique pour PRISM CINE V2.

Remplit l'UI avec des données réalistes pour une démo projet/soutenance :
- Reset système
- Génère 60-80 ratings réalistes (10-15 par utilisateur)
- Crée des profils utilisateurs avec biais de genre
- Vérifie convergence SVD (MATCH vs DISCOVERY)
- Génère un rapport textuel

Usage:
    python demo_fill.py --users 5 --ratings 80
"""

import requests
import time
import random
import argparse
from typing import Dict, List

BASE_URL = "http://localhost:5000"

# ============== User Profiles (genre biases) ==============
USER_PROFILES = {
    "aymen": {
        "preferred_genres": ["Sci-Fi", "Thriller"],
        "rating_bias": 0.8,  # 80% films préférés
    },
    "ichrak": {
        "preferred_genres": ["Horror", "Thriller"],
        "rating_bias": 0.7,
    },
    "oualid": {
        "preferred_genres": ["Action", "Sci-Fi"],
        "rating_bias": 0.75,
    },
    "sara": {
        "preferred_genres": ["Drama", "Romance"],
        "rating_bias": 0.75,
    },
    "mohammed": {
        "preferred_genres": ["Comedy", "Animation"],
        "rating_bias": 0.65,
    },
    "rachid": {
        "preferred_genres": ["Fantasy", "Adventure"],
        "rating_bias": 0.7,
    },
    "youssef": {
        "preferred_genres": ["Horror", "Thriller"],
        "rating_bias": 0.7,
    },
    "lina": {
        "preferred_genres": ["Drama", "Romance"],
        "rating_bias": 0.75,
    },
    "nassim": {
        "preferred_genres": ["Comedy", "Animation"],
        "rating_bias": 0.65,
    },
    "amina": {
        "preferred_genres": ["Fantasy", "Adventure"],
        "rating_bias": 0.7,
    },
}

# ============== API Helpers ==============
def reset_system():
    """Reset complet du système."""
    print("\n🔄 RESET SYSTÈME...")
    try:
        resp = requests.post(f"{BASE_URL}/api/reset", timeout=5)
        data = resp.json()
        print(f"   ✓ {data['deleted_users']} users + {data['deleted_ratings']} ratings supprimés")
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"   ✗ Erreur reset: {e}")
        return False

def get_all_movies() -> List[Dict]:
    """Récupère la liste de tous les films."""
    # Hardcoder movie_1 à movie_50 pour simplifier
    return [{"movie_id": f"movie_{i}"} for i in range(1, 51)]

def rate_movie(user_id: str, movie_id: str, rating: float) -> Dict:
    """Envoie un rating."""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/rate",
            json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
            timeout=5
        )
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_recommendations(user_id: str, n: int = 8) -> Dict:
    """Récupère les recommandations."""
    try:
        resp = requests.get(f"{BASE_URL}/api/recommend/{user_id}", params={"n": n}, timeout=5)
        return resp.json()
    except Exception as e:
        return {"recommendations": []}

def get_stats() -> Dict:
    """Récupère les stats système."""
    try:
        resp = requests.get(f"{BASE_URL}/api/stats", timeout=5)
        return resp.json()
    except Exception as e:
        return {}

# ============== Data Generation ==============
def generate_ratings(users: List[str], movies: List[Dict], total_ratings: int):
    """
    Génère des ratings réalistes avec biais de genre.
    """
    print(f"\n📊 GÉNÉRATION DE {total_ratings} RATINGS...")
    print(f"   Users: {', '.join(users)}")

    # Distribuer les ratings équitablement
    ratings_per_user = total_ratings // len(users)

    # Mapping genre par film (hardcodé pour simplifier)
    genre_map = {
        **{f"movie_{i}": "Action" for i in range(1, 6)},
        **{f"movie_{i}": "Sci-Fi" for i in range(6, 11)},
        **{f"movie_{i}": "Drama" for i in range(11, 16)},
        **{f"movie_{i}": "Romance" for i in range(16, 21)},
        **{f"movie_{i}": "Animation" for i in range(21, 26)},
        **{f"movie_{i}": "Thriller" for i in range(26, 31)},
        **{f"movie_{i}": "Horror" for i in range(31, 36)},
        **{f"movie_{i}": "Comedy" for i in range(36, 41)},
        **{f"movie_{i}": "Fantasy" for i in range(41, 46)},
        **{f"movie_{i}": "Adventure" for i in range(46, 51)},
    }

    for user in users:
        profile = USER_PROFILES.get(user, {"preferred_genres": [], "rating_bias": 0.5})
        preferred = profile["preferred_genres"]
        bias = profile["rating_bias"]

        print(f"\n   👤 {user.upper()} (préfère: {', '.join(preferred)})")

        # Sélectionner films préférés
        preferred_movies = [
            m for m in movies if genre_map.get(m["movie_id"]) in preferred
        ]
        other_movies = [
            m for m in movies if genre_map.get(m["movie_id"]) not in preferred
        ]

        # Ratio preferred / other selon bias
        n_preferred = int(ratings_per_user * bias)
        n_other = ratings_per_user - n_preferred

        # IMPORTANT : laisser au moins 3 films préférés NON-NOTÉS
        # pour que le recommender puisse les suggérer en MATCH.
        leave_for_reco = 3
        max_preferred = max(1, len(preferred_movies) - leave_for_reco)
        n_preferred = min(n_preferred, max_preferred)
        n_other = ratings_per_user - n_preferred

        # Sélectionner aléatoirement
        selected_preferred = random.sample(preferred_movies, min(n_preferred, len(preferred_movies)))
        selected_other = random.sample(other_movies, min(n_other, len(other_movies)))

        # Noter avec scores appropriés
        for movie in selected_preferred:
            rating = round(random.uniform(4.0, 5.0), 1)  # Haute note pour préférés
            result = rate_movie(user, movie["movie_id"], rating)
            if result.get("status") == "ok":
                print(f"      ✓ {movie['movie_id']} = {rating}")
            time.sleep(0.05)  # Délai réaliste

        for movie in selected_other:
            rating = round(random.uniform(2.5, 4.0), 1)  # Note moyenne pour autres
            result = rate_movie(user, movie["movie_id"], rating)
            if result.get("status") == "ok":
                print(f"      · {movie['movie_id']} = {rating}")
            time.sleep(0.05)

# ============== Verification ==============
def verify_recommendations(users: List[str]):
    """
    Vérifie que chaque utilisateur a des recommendations personnalisées.
    """
    print(f"\n🔍 VÉRIFICATION RECOMMANDATIONS...")

    for user in users:
        recs_data = get_recommendations(user, n=8)
        recs = recs_data.get("recommendations", [])

        if not recs:
            print(f"\n   👤 {user.upper()}: Aucune recommandation")
            continue

        match_count = sum(1 for r in recs if r.get("strategy") == "MATCH")
        discovery_count = sum(1 for r in recs if r.get("strategy") == "DISCOVERY")

        avg_score = sum(r.get("predicted_score", 0) for r in recs) / len(recs) if recs else 0

        print(f"\n   👤 {user.upper()}:")
        print(f"      MATCH: {match_count}/8 ({match_count/8*100:.0f}%)")
        print(f"      DISCOVERY: {discovery_count}/8 ({discovery_count/8*100:.0f}%)")
        print(f"      Score moyen: {avg_score:.2f}/5")

        # Afficher top-3
        for i, rec in enumerate(recs[:3], 1):
            badge = "🎲" if rec.get("strategy") == "DISCOVERY" else "✓"
            title = rec.get("title", "Unknown")
            genre = rec.get("genre", "N/A")
            score = rec.get("predicted_score", 0)
            print(f"      {i}. {badge} {title} ({genre}) - {score:.2f}")

def print_final_stats():
    """Affiche les stats finales."""
    print(f"\n📈 STATISTIQUES FINALES")
    stats = get_stats()

    if not stats:
        print("   Erreur: impossible de récupérer les stats")
        return

    print(f"   Total users: {stats.get('total_users', 0)}")
    print(f"   Total ratings: {stats.get('total_ratings', 0)}")
    print(f"   Total movies: {stats.get('total_movies', 0)}")

    total_users = stats.get('total_users', 0)
    total_movies = stats.get('total_movies', 0)
    total_ratings = stats.get('total_ratings', 0)

    if total_users > 0 and total_movies > 0:
        density = (total_ratings / (total_users * total_movies)) * 100
        print(f"   Matrix density: {density:.1f}%")

    genre_pop = stats.get('genre_popularity', {})
    if genre_pop:
        print(f"\n   Genre popularity:")
        for genre, pop in sorted(genre_pop.items(), key=lambda x: x[1], reverse=True):
            print(f"      {genre}: {pop*100:.1f}%")

# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="PRISM CINE V2 Demo Fill Script")
    parser.add_argument("--users", type=int, default=5, help="Number of users (default: 5)")
    parser.add_argument("--ratings", type=int, default=None, help="Total ratings (default: auto-calculated for 32%% density)")
    parser.add_argument("--density", type=float, default=0.32, help="Target matrix density (default: 0.32)")
    args = parser.parse_args()

    # Auto-calculate ratings based on target density if not specified
    num_movies = 50  # Fixed number of movies in the catalog
    if args.ratings is None:
        args.ratings = int(args.density * args.users * num_movies)

    print("=" * 75)
    print(" KINETOSCOPE V2 — DÉMONSTRATION AUTOMATIQUE ".center(75))
    print("=" * 75)
    print(f"\n🎯 Configuration: {args.users} users, {args.ratings} ratings")
    print(f"   Density cible: {(args.ratings / (args.users * num_movies)) * 100:.1f}%\n")

    # Phase 1: Reset
    if not reset_system():
        print("✗ Impossible de contacter le serveur. Vérifiez que docker-compose up fonctionne.")
        return

    # Phase 2: Generate data
    users = list(USER_PROFILES.keys())[:args.users]
    movies = get_all_movies()
    generate_ratings(users, movies, args.ratings)

    # Phase 3: Verify
    time.sleep(1)
    verify_recommendations(users)

    # Phase 4: Stats
    print_final_stats()

    print("\n" + "=" * 75)
    print(" DÉMONSTRATION COMPLÈTE".center(75))
    print(f"Ouvrir http://localhost:5000".center(75))
    print("=" * 75)

if __name__ == "__main__":
    main()
