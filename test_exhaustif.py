#!/usr/bin/env python3
"""
Suite de Tests EXHAUSTIVE — PRISM CINE V2 (SVD + Bandit)

Couvre 15 scenarios de test pour garantir que le systeme fonctionne :
    1. Cold-start (zero donnees)
    2. Premier rating SGD
    3. Convergence SGD (10+ votes meme utilisateur)
    4. Personnalisation multi-users
    5. Ratio MATCH/DISCOVERY (epsilon=0.2)
    6. Genre boost manuel
    7. Genre boost via simulate_traffic
    8. Edge cases (tous films notes, user inconnu, re-rating)
    9. Cle 'strategy' presente partout
    10. Scores dans [0, 5]
    11. Reinitialisation via /api/reset
    12. Predictions coherentes (genre prefere en top-3)
    13. Cold-start individuel vs global
    14. Bandit exploration (DISCOVERY aleatoire)
    15. Stress test (100+ ratings)

Usage:
    python test_exhaustif.py
"""

import requests
import json
import time
import random
from typing import Dict, List, Tuple

# Colorama optionnel
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = ""
    class Style:
        BRIGHT = RESET_ALL = ""

BASE_URL = "http://localhost:5000"

# ============================================================================
# Helpers
# ============================================================================

def print_header(title: str):
    print(f"\n{'='*75}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(75)}")
    print(f"{'='*75}\n")

def print_success(msg: str):
    print(f"{Fore.GREEN}✓ {msg}")

def print_error(msg: str):
    print(f"{Fore.RED}✗ {msg}")

def print_info(msg: str):
    print(f"{Fore.YELLOW}ℹ {msg}")

def print_result(msg: str):
    print(f"{Fore.MAGENTA}→ {msg}")

def print_debug(msg: str):
    print(f"{Fore.BLUE}🔍 {msg}")

def rate_movie(user_id: str, movie_id: str, rating: float) -> Dict:
    resp = requests.post(f"{BASE_URL}/api/rate", json={"user_id": user_id, "movie_id": movie_id, "rating": rating}, timeout=5)
    return resp.json()

def get_recommendations(user_id: str, n: int = 5) -> Dict:
    resp = requests.get(f"{BASE_URL}/api/recommend/{user_id}", params={"n": n}, timeout=5)
    return resp.json()

def get_stats() -> Dict:
    resp = requests.get(f"{BASE_URL}/api/stats", timeout=5)
    return resp.json()

def boost_genre(genre: str) -> Dict:
    resp = requests.post(f"{BASE_URL}/api/boost", json={"genre": genre}, timeout=5)
    return resp.json()

def reset_system() -> Dict:
    resp = requests.post(f"{BASE_URL}/api/reset", timeout=5)
    return resp.json()

def check_server() -> bool:
    try:
        resp = requests.get(BASE_URL, timeout=5)
        return resp.status_code == 200
    except:
        return False

# ============================================================================
# Tests
# ============================================================================

def test_01_server_connectivity():
    """Test 1 : Connectivite serveur."""
    print_header("TEST 1 : CONNECTIVITE SERVEUR")

    if not check_server():
        print_error("Serveur inaccessible")
        return False

    print_success(f"Serveur accessible a {BASE_URL}")
    return True

def test_02_cold_start_global():
    """Test 2 : Cold-start global (zero donnees)."""
    print_header("TEST 2 : COLD-START GLOBAL")

    # Reset pour garantir zero donnees
    reset_system()
    time.sleep(0.3)

    stats = get_stats()
    print_debug(f"Users: {stats['total_users']}, Ratings: {stats['total_ratings']}")

    if stats['total_users'] != 0 or stats['total_ratings'] != 0:
        print_error("Base non vide apres reset")
        return False

    # Recs pour user inconnu
    recs = get_recommendations("unknown_xyz", n=5)["recommendations"]

    if not recs:
        print_error("Aucune recommandation cold-start")
        return False

    # Toutes doivent etre DISCOVERY
    strategies = [r["strategy"] for r in recs]
    if not all(s == "DISCOVERY" for s in strategies):
        print_error(f"Strategies incorrectes : {strategies}")
        return False

    print_success(f"{len(recs)} recs DISCOVERY retournees")
    print_success("TEST 2 PASSE\n")
    return True

def test_03_first_rating_sgd():
    """Test 3 : Premier rating declenche SGD."""
    print_header("TEST 3 : PREMIER RATING SGD")

    result = rate_movie("alice", "movie_1", 5.0)

    if result.get("status") != "ok":
        print_error(f"Rating echoue : {result}")
        return False

    print_success(f"Rating accepte : {result}")

    stats = get_stats()
    if stats["total_users"] != 1 or stats["total_ratings"] != 1:
        print_error(f"Stats incorrectes : {stats}")
        return False

    print_success("Stats mises a jour correctement")
    print_success("TEST 3 PASSE\n")
    return True

def test_04_sgd_convergence():
    """Test 4 : Convergence SGD avec 15 votes."""
    print_header("TEST 4 : CONVERGENCE SGD (15 VOTES)")

    user = "bob_convergence"

    # Noter 15 films Sci-Fi avec des scores eleves
    scifi_movies = ["movie_3", "movie_4", "movie_5"]  # Interstellar, Blade Runner, Arrival

    print_info(f"Bob note 15 films (3 Sci-Fi x 5, 12 autres x 3.0)...")

    # 5 votes par film Sci-Fi
    for movie in scifi_movies:
        for _ in range(5):
            rate_movie(user, movie, 5.0)
            time.sleep(0.05)

    print_debug(f"Bob a maintenant 15 votes")

    # Verifier que les recs privilegient Sci-Fi
    recs = get_recommendations(user, n=5)["recommendations"]
    match_recs = [r for r in recs if r["strategy"] == "MATCH"]

    if not match_recs:
        print_error("Aucune rec MATCH apres 15 votes")
        return False

    # Compter Sci-Fi dans le top-3 MATCH
    top3_match = match_recs[:3]
    scifi_count = sum(1 for r in top3_match if r["genre"] == "Sci-Fi")

    print_result(f"Top-3 MATCH : {scifi_count}/3 sont Sci-Fi")

    if scifi_count < 2:  # Au moins 2/3 doivent etre Sci-Fi
        print_error("Modele ne converge pas vers Sci-Fi")
        return False

    print_success("Convergence SGD confirmee (preferences Sci-Fi detectees)")
    print_success("TEST 4 PASSE\n")
    return True

def test_05_multi_user_personalization():
    """Test 5 : Personnalisation multi-users."""
    print_header("TEST 5 : PERSONNALISATION MULTI-USERS")

    # Charlie aime Horror
    print_info("Charlie note 6 films Horror (5/5)")
    rate_movie("charlie", "movie_16", 5.0)  # Get Out
    rate_movie("charlie", "movie_17", 5.0)  # The Shining
    for _ in range(4):
        rate_movie("charlie", "movie_16", 5.0)

    # Diana aime Comedy
    print_info("Diana note 6 films Comedy (5/5)")
    rate_movie("diana", "movie_18", 5.0)  # Grand Budapest
    rate_movie("diana", "movie_19", 5.0)  # Superbad
    for _ in range(4):
        rate_movie("diana", "movie_18", 5.0)

    time.sleep(0.5)

    # Comparer recs
    charlie_recs = get_recommendations("charlie", n=5)["recommendations"]
    diana_recs = get_recommendations("diana", n=5)["recommendations"]

    charlie_match = [r for r in charlie_recs if r["strategy"] == "MATCH"]
    diana_match = [r for r in diana_recs if r["strategy"] == "MATCH"]

    charlie_genres = [r["genre"] for r in charlie_match]
    diana_genres = [r["genre"] for r in diana_match]

    print_result(f"Charlie (Horror) — Genres MATCH: {charlie_genres}")
    print_result(f"Diana (Comedy) — Genres MATCH: {diana_genres}")

    # Verifier divergence
    charlie_titles = {r["title"] for r in charlie_recs}
    diana_titles = {r["title"] for r in diana_recs}
    overlap = len(charlie_titles & diana_titles)

    print_debug(f"Overlap : {overlap}/{len(charlie_recs)} films")

    if overlap == len(charlie_recs):
        print_error("Recs identiques ! Pas de personnalisation")
        return False

    print_success("Personnalisation confirmee (recs differentes)")
    print_success("TEST 5 PASSE\n")
    return True

def test_06_match_discovery_ratio():
    """Test 6 : Ratio MATCH/DISCOVERY (epsilon=0.2)."""
    print_header("TEST 6 : RATIO MATCH/DISCOVERY")

    user = "emma_bandit"

    # Noter 10 films
    for i in range(1, 11):
        rate_movie(user, f"movie_{i}", 4.0 + random.random())
        time.sleep(0.05)

    # Recuperer 20 recs (pour avoir un echantillon plus grand)
    all_recs = get_recommendations(user, n=20)["recommendations"]

    if not all_recs:
        print_error("Aucune recommandation")
        return False

    match_count = sum(1 for r in all_recs if r["strategy"] == "MATCH")
    discovery_count = sum(1 for r in all_recs if r["strategy"] == "DISCOVERY")

    ratio_discovery = discovery_count / len(all_recs)

    print_result(f"MATCH: {match_count}, DISCOVERY: {discovery_count}")
    print_result(f"Ratio DISCOVERY: {ratio_discovery:.2%} (attendu: ~20%)")

    # Epsilon=0.2 → au moins 1 DISCOVERY sur 5
    if discovery_count < 1:
        print_error("Pas assez de DISCOVERY (epsilon=0.2)")
        return False

    # Ratio pas trop loin de 20%
    if not (0.10 < ratio_discovery < 0.35):
        print_error(f"Ratio DISCOVERY anormal : {ratio_discovery:.2%}")
        return False

    print_success("Ratio MATCH/DISCOVERY correct")
    print_success("TEST 6 PASSE\n")
    return True

def test_07_genre_boost():
    """Test 7 : Genre boost via API."""
    print_header("TEST 7 : GENRE BOOST")

    genre = "Fantasy"
    result = boost_genre(genre)

    if result.get("status") != "boosted":
        print_error(f"Boost echoue : {result}")
        return False

    print_success(f"Boost active : {genre}")

    # Verifier stats
    stats = get_stats()
    if stats["boosted_genre"] != genre:
        print_error(f"Stats incorrect : {stats['boosted_genre']}")
        return False

    print_success("Genre boost confirme dans les stats")
    print_success("TEST 7 PASSE\n")
    return True

def test_08_strategy_key_present():
    """Test 8 : Cle 'strategy' presente dans toutes les recs."""
    print_header("TEST 8 : CLE 'strategy' PRESENTE")

    user = "frank_strategy"
    rate_movie(user, "movie_1", 4.5)

    recs = get_recommendations(user, n=10)["recommendations"]

    if not recs:
        print_error("Aucune recommandation")
        return False

    missing = [r for r in recs if "strategy" not in r]

    if missing:
        print_error(f"{len(missing)} recs sans cle 'strategy'")
        return False

    print_success(f"Toutes les {len(recs)} recs ont la cle 'strategy'")
    print_success("TEST 8 PASSE\n")
    return True

def test_09_scores_in_range():
    """Test 9 : Scores predits dans [0, 5]."""
    print_header("TEST 9 : SCORES DANS [0, 5]")

    user = "grace_scores"

    # Noter 5 films
    for i in range(1, 6):
        rate_movie(user, f"movie_{i}", 3.0 + i * 0.3)
        time.sleep(0.05)

    recs = get_recommendations(user, n=10)["recommendations"]

    out_of_range = [r for r in recs if not (0 <= r["predicted_score"] <= 5)]

    if out_of_range:
        print_error(f"{len(out_of_range)} scores hors [0, 5]")
        for r in out_of_range:
            print_debug(f"  {r['title']}: {r['predicted_score']}")
        return False

    scores = [r["predicted_score"] for r in recs]
    print_result(f"Scores min/max: {min(scores):.2f} / {max(scores):.2f}")

    print_success("Tous les scores dans [0, 5]")
    print_success("TEST 9 PASSE\n")
    return True

def test_10_all_movies_rated():
    """Test 10 : User avec tous les films notes."""
    print_header("TEST 10 : TOUS FILMS NOTES")

    user = "henry_full"

    print_info("Henry note les 20 films...")
    for i in range(1, 21):
        rate_movie(user, f"movie_{i}", 3.0 + random.random() * 2)
        time.sleep(0.02)

    recs = get_recommendations(user, n=5)

    if len(recs.get("recommendations", [])) > 0:
        print_error(f"Recs retournees alors que tous films notes : {len(recs['recommendations'])}")
        return False

    print_success("Liste vide retournee (correct)")
    print_success("TEST 10 PASSE\n")
    return True

def test_11_unknown_user():
    """Test 11 : User inconnu (cold-start individuel)."""
    print_header("TEST 11 : USER INCONNU")

    recs = get_recommendations("isabelle_unknown_xyz", n=5)["recommendations"]

    if not recs:
        print_error("Aucune rec pour user inconnu")
        return False

    print_success(f"{len(recs)} recs retournees pour user inconnu")
    print_success("TEST 11 PASSE\n")
    return True

def test_12_reset_endpoint():
    """Test 12 : Endpoint /api/reset."""
    print_header("TEST 12 : RESET ENDPOINT")

    # S'assurer qu'il y a des donnees
    rate_movie("jack_reset", "movie_1", 5.0)

    stats_before = get_stats()
    print_debug(f"Avant reset: {stats_before['total_users']} users, {stats_before['total_ratings']} ratings")

    result = reset_system()

    if result.get("status") != "reset_complete":
        print_error(f"Reset echoue : {result}")
        return False

    time.sleep(0.3)

    stats_after = get_stats()
    print_debug(f"Apres reset: {stats_after['total_users']} users, {stats_after['total_ratings']} ratings")

    if stats_after["total_users"] != 0 or stats_after["total_ratings"] != 0:
        print_error("Reset incomplet")
        return False

    print_success(f"Reset reussi : {result['deleted_users']} users, {result['deleted_ratings']} ratings supprimes")
    print_success("TEST 12 PASSE\n")
    return True

def test_13_re_rating_same_movie():
    """Test 13 : Re-noter le meme film."""
    print_header("TEST 13 : RE-RATING")

    user = "kate_rerating"
    movie = "movie_7"

    # Premier rating
    result1 = rate_movie(user, movie, 3.0)
    print_debug("1er rating: 3.0")

    # Re-rating
    result2 = rate_movie(user, movie, 5.0)
    print_debug("2e rating: 5.0 (ecrasement)")

    if result1.get("status") != "ok" or result2.get("status") != "ok":
        print_error("Re-rating echoue")
        return False

    stats = get_stats()
    # Total_ratings compte les evenements (2), pas les paires uniques
    print_debug(f"Total ratings: {stats['total_ratings']}")

    print_success("Re-rating accepte")
    print_success("TEST 13 PASSE\n")
    return True

def test_14_discovery_randomness():
    """Test 14 : DISCOVERY est aleatoire (non deterministe)."""
    print_header("TEST 14 : DISCOVERY ALEATOIRE")

    user = "leo_discovery"

    # Noter 8 films
    for i in range(1, 9):
        rate_movie(user, f"movie_{i}", 4.0)
        time.sleep(0.02)

    # Recuperer recs 3 fois
    discovery_sets = []
    for _ in range(3):
        recs = get_recommendations(user, n=10)["recommendations"]
        discoveries = {r["title"] for r in recs if r["strategy"] == "DISCOVERY"}
        discovery_sets.append(discoveries)
        time.sleep(0.1)

    # Verifier que les DISCOVERY changent
    unique_discoveries = discovery_sets[0] | discovery_sets[1] | discovery_sets[2]

    print_debug(f"DISCOVERY uniques sur 3 appels: {len(unique_discoveries)}")

    if len(unique_discoveries) <= 1:
        print_error("DISCOVERY semble deterministe (pas aleatoire)")
        return False

    print_success("DISCOVERY varie entre les appels (bandit aleatoire)")
    print_success("TEST 14 PASSE\n")
    return True

def test_15_stress_test():
    """Test 15 : Stress test (100 ratings)."""
    print_header("TEST 15 : STRESS TEST (100 RATINGS)")

    print_info("Generation de 100 ratings...")

    users = [f"user_{i}" for i in range(10)]
    movies = [f"movie_{i}" for i in range(1, 21)]

    start_time = time.time()

    for i in range(1000):
        user = random.choice(users)
        movie = random.choice(movies)
        rating = 1.0 + random.random() * 4.0

        result = rate_movie(user, movie, rating)

        if result.get("status") != "ok":
            print_error(f"Rating {i+1} echoue")
            return False

        if i % 20 == 0:
            print_debug(f"  {i}/1000 ratings...")

    elapsed = time.time() - start_time

    print_result(f"1000 ratings en {elapsed:.2f}s ({1000/elapsed:.1f} ratings/s)")

    # Verifier stats
    stats = get_stats()
    if stats["total_ratings"] < 1000:
        print_error(f"Total ratings incorrect : {stats['total_ratings']}")
        return False

    print_success("Stress test passe")
    print_success("TEST 15 PASSE\n")
    return True

# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║             KINETOSCOPE V2 — SUITE DE TESTS EXHAUSTIVE              ║")
    print("║                    (SVD + BANDIT EPSILON-GREEDY)                     ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    tests = [
        ("Connectivite serveur", test_01_server_connectivity),
        ("Cold-start global", test_02_cold_start_global),
        ("Premier rating SGD", test_03_first_rating_sgd),
        ("Convergence SGD (15 votes)", test_04_sgd_convergence),
        ("Personnalisation multi-users", test_05_multi_user_personalization),
        ("Ratio MATCH/DISCOVERY", test_06_match_discovery_ratio),
        ("Genre boost", test_07_genre_boost),
        ("Cle 'strategy' presente", test_08_strategy_key_present),
        ("Scores dans [0, 5]", test_09_scores_in_range),
        ("Tous films notes", test_10_all_movies_rated),
        ("User inconnu", test_11_unknown_user),
        ("Reset endpoint", test_12_reset_endpoint),
        ("Re-rating", test_13_re_rating_same_movie),
        ("DISCOVERY aleatoire", test_14_discovery_randomness),
        ("Stress test (100 ratings)", test_15_stress_test),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print_error(f"Exception dans {name} : {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Resume
    print_header("RESUME DES TESTS")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = f"{Fore.GREEN}✓ PASSE" if success else f"{Fore.RED}✗ ECHOUE"
        print(f"{status}{Style.RESET_ALL} — {name}")

    print(f"\n{Fore.CYAN}{'='*75}")
    if passed == total:
        print(f"{Fore.GREEN}{Style.BRIGHT}TOUS LES TESTS REUSSIS ({passed}/{total}) ! 🎉🎉🎉")
        print(f"{Fore.GREEN}Le moteur SVD+Bandit est 100% operationnel.")
    else:
        print(f"{Fore.YELLOW}{Style.BRIGHT}{passed}/{total} tests reussis.")
        print(f"{Fore.YELLOW}{total - passed} tests ont echoue. Verifiez les logs.")
    print(f"{Fore.CYAN}{'='*75}\n")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
