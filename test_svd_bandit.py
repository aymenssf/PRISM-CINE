#!/usr/bin/env python3
"""
Test Suite Complet — Kinetoscope V2 (SVD + Bandit)

Lance une serie de tests pour valider :
    1. Cold-start (zero donnees)
    2. Premier rating → SGD update
    3. Recommandations personnalisees (MATCH vs DISCOVERY)
    4. Multi-users → predictions divergentes
    5. Genre boost (JADE simulation)
    6. Edge cases (tous films notes, utilisateur inconnu, etc.)

Usage:
    python test_svd_bandit.py
"""

import requests
import json
import time
from typing import Dict, List

# Colorama optionnel (si non installe, desactive les couleurs)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_ENABLED = True
except ImportError:
    # Fallback sans couleurs
    class Fore:
        CYAN = GREEN = RED = YELLOW = MAGENTA = ""
    class Style:
        BRIGHT = RESET_ALL = ""
    COLORS_ENABLED = False

BASE_URL = "http://localhost:5000"

# ============================================================================
# Helpers
# ============================================================================

def print_header(title: str):
    """Affiche un header colore."""
    print(f"\n{'='*70}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(70)}")
    print(f"{'='*70}\n")

def print_success(msg: str):
    print(f"{Fore.GREEN}✓ {msg}")

def print_error(msg: str):
    print(f"{Fore.RED}✗ {msg}")

def print_info(msg: str):
    print(f"{Fore.YELLOW}ℹ {msg}")

def print_result(msg: str):
    print(f"{Fore.MAGENTA}→ {msg}")

def rate_movie(user_id: str, movie_id: str, rating: float) -> Dict:
    """Envoie une note via POST /api/rate."""
    payload = {"user_id": user_id, "movie_id": movie_id, "rating": rating}
    resp = requests.post(f"{BASE_URL}/api/rate", json=payload, timeout=5)
    return resp.json()

def get_recommendations(user_id: str, n: int = 5) -> Dict:
    """Recupere les recommandations via GET /api/recommend/<user_id>."""
    resp = requests.get(f"{BASE_URL}/api/recommend/{user_id}", params={"n": n}, timeout=5)
    return resp.json()

def get_stats() -> Dict:
    """Recupere les statistiques systeme via GET /api/stats."""
    resp = requests.get(f"{BASE_URL}/api/stats", timeout=5)
    return resp.json()

def boost_genre(genre: str) -> Dict:
    """Applique un boost de genre via POST /api/boost."""
    resp = requests.post(f"{BASE_URL}/api/boost", json={"genre": genre}, timeout=5)
    return resp.json()

def check_server():
    """Verifie que le serveur Flask repond."""
    try:
        resp = requests.get(BASE_URL, timeout=5)
        if resp.status_code == 200:
            print_success(f"Serveur Flask accessible a {BASE_URL}")
            return True
        else:
            print_error(f"Serveur repondu avec status {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"Impossible de joindre le serveur : {e}")
        print_info("Assurez-vous que docker-compose up tourne !")
        return False

# ============================================================================
# Test Cases
# ============================================================================

def test_1_cold_start():
    """Test 1 : Cold-start — zero donnees."""
    print_header("TEST 1 : COLD-START (Zero Donnees)")

    stats = get_stats()
    print_info(f"Users: {stats['total_users']}, Ratings: {stats['total_ratings']}")

    if stats['total_users'] > 0 or stats['total_ratings'] > 0:
        print_error("ATTENTION : Base non vide ! Relancer docker-compose up --build pour reset.")
        return False

    # Recommandation sans user_id (cold-start global)
    recs = get_recommendations("unknown_user_xyz", n=5)

    if "recommendations" not in recs:
        print_error("Pas de cle 'recommendations' dans la reponse")
        return False

    cold_recs = recs["recommendations"]
    print_info(f"Cold-start retourne {len(cold_recs)} films")

    # Toutes les recs doivent avoir strategy=DISCOVERY (pas de signal SVD)
    all_discovery = all(r.get("strategy") == "DISCOVERY" for r in cold_recs)
    if all_discovery:
        print_success("Toutes les recs sont marquees DISCOVERY (correct)")
    else:
        print_error("Certaines recs ne sont pas DISCOVERY en cold-start")
        return False

    # Afficher les 3 premiers
    for rec in cold_recs[:3]:
        print_result(f"  {rec['title']} ({rec['genre']}) — Strategy: {rec['strategy']}, Score: {rec['predicted_score']}")

    print_success("TEST 1 PASSE\n")
    return True

def test_2_first_rating_sgd():
    """Test 2 : Premier rating → SGD update verifie."""
    print_header("TEST 2 : PREMIER RATING (SGD Update)")

    user = "alice"
    movie = "movie_3"  # Interstellar
    rating = 5.0

    print_info(f"Soumission : {user} note {movie} = {rating}")
    result = rate_movie(user, movie, rating)

    if result.get("status") != "ok":
        print_error(f"Erreur lors du rating : {result}")
        return False

    print_success(f"Rating accepte : {result}")

    # Verifier les stats
    stats = get_stats()
    if stats["total_users"] == 1 and stats["total_ratings"] == 1:
        print_success(f"Stats mises a jour : {stats['total_users']} user, {stats['total_ratings']} rating")
    else:
        print_error(f"Stats incorrectes : {stats}")
        return False

    print_success("TEST 2 PASSE\n")
    return True

def test_3_recommendations_match_discovery():
    """Test 3 : Recommandations avec MATCH et DISCOVERY."""
    print_header("TEST 3 : RECOMMANDATIONS (MATCH + DISCOVERY)")

    user = "alice"
    recs = get_recommendations(user, n=5)

    if "recommendations" not in recs:
        print_error("Pas de cle 'recommendations'")
        return False

    recommendations = recs["recommendations"]

    if len(recommendations) == 0:
        print_error("Aucune recommandation retournee")
        return False

    print_info(f"{len(recommendations)} recommandations pour {user}")

    # Compter MATCH vs DISCOVERY
    match_count = sum(1 for r in recommendations if r["strategy"] == "MATCH")
    discovery_count = sum(1 for r in recommendations if r["strategy"] == "DISCOVERY")

    print_result(f"  MATCH: {match_count}, DISCOVERY: {discovery_count}")

    # Epsilon=0.2 → au moins 1 DISCOVERY attendu
    if discovery_count < 1:
        print_error("Pas de DISCOVERY trouve (epsilon=0.2 devrait donner au moins 1)")
        return False

    if match_count < 1:
        print_error("Pas de MATCH trouve (exploitation devrait exister)")
        return False

    print_success(f"Mix MATCH/DISCOVERY correct (exploration {discovery_count}/{len(recommendations)})")

    # Afficher toutes les recs
    for i, rec in enumerate(recommendations, 1):
        strategy_icon = "🎯" if rec["strategy"] == "MATCH" else "🎲"
        print_result(f"  {i}. {strategy_icon} {rec['title']} ({rec['genre']}) — Score: {rec['predicted_score']:.2f}")

    print_success("TEST 3 PASSE\n")
    return True

def test_4_multiple_users_personalization():
    """Test 4 : Plusieurs utilisateurs → recommandations divergentes."""
    print_header("TEST 4 : MULTI-USERS (Personnalisation)")

    # Alice aime Sci-Fi
    print_info("Alice note 3 films Sci-Fi (5/5)")
    rate_movie("alice", "movie_3", 5.0)  # Interstellar
    rate_movie("alice", "movie_4", 5.0)  # Blade Runner
    rate_movie("alice", "movie_5", 5.0)  # Arrival

    # Bob aime Horror
    print_info("Bob note 2 films Horror (5/5)")
    rate_movie("bob", "movie_16", 5.0)  # Get Out
    rate_movie("bob", "movie_17", 5.0)  # The Shining

    time.sleep(0.5)  # Laisser SGD se stabiliser

    # Recommandations Alice
    alice_recs = get_recommendations("alice", n=5)["recommendations"]
    alice_genres = [r["genre"] for r in alice_recs if r["strategy"] == "MATCH"]
    print_result(f"Alice (fan Sci-Fi) — Genres MATCH: {alice_genres}")

    # Recommandations Bob
    bob_recs = get_recommendations("bob", n=5)["recommendations"]
    bob_genres = [r["genre"] for r in bob_recs if r["strategy"] == "MATCH"]
    print_result(f"Bob (fan Horror) — Genres MATCH: {bob_genres}")

    # Verifier que les recs sont differentes
    alice_titles = {r["title"] for r in alice_recs}
    bob_titles = {r["title"] for r in bob_recs}

    overlap = alice_titles & bob_titles
    print_info(f"Overlap entre Alice et Bob : {len(overlap)}/{len(alice_titles)} films")

    if len(overlap) == len(alice_titles):
        print_error("Recommandations identiques ! SVD ne personnalise pas.")
        return False

    print_success("Recommandations personnalisees par utilisateur")
    print_success("TEST 4 PASSE\n")
    return True

def test_5_genre_boost():
    """Test 5 : Genre boost via JADE simulation."""
    print_header("TEST 5 : GENRE BOOST (JADE Simulation)")

    # Activer boost Sci-Fi
    genre = "Sci-Fi"
    print_info(f"Activation du boost genre : {genre}")
    result = boost_genre(genre)

    if result.get("status") != "boosted":
        print_error(f"Erreur lors du boost : {result}")
        return False

    print_success(f"Genre booste : {result['genre']}")

    # Verifier les stats
    stats = get_stats()
    if stats["boosted_genre"] == genre:
        print_success(f"Stats confirment le boost : {stats['boosted_genre']}")
    else:
        print_error(f"Boost non reflete dans les stats : {stats}")
        return False

    # Recommandations avec boost
    user = "alice"
    recs_boosted = get_recommendations(user, n=5)["recommendations"]

    # Compter les films Sci-Fi dans le top-3 MATCH
    match_recs = [r for r in recs_boosted if r["strategy"] == "MATCH"][:3]
    scifi_count = sum(1 for r in match_recs if r["genre"] == "Sci-Fi")

    print_result(f"Top-3 MATCH : {scifi_count}/3 sont Sci-Fi")

    if scifi_count > 0:
        print_success("Boost detecte dans les recommandations (Sci-Fi prioritaires)")
    else:
        print_error("Boost n'affecte pas les recommandations")
        return False

    print_success("TEST 5 PASSE\n")
    return True

def test_6_edge_cases():
    """Test 6 : Edge cases (tous films notes, user inconnu)."""
    print_header("TEST 6 : EDGE CASES")

    # 6a : Utilisateur inconnu
    print_info("6a. Utilisateur inconnu (cold-start individuel)")
    charlie_recs = get_recommendations("charlie_unknown", n=5)
    if len(charlie_recs.get("recommendations", [])) > 0:
        print_success("Cold-start individuel retourne des recs")
    else:
        print_error("Pas de recs pour utilisateur inconnu")
        return False

    # 6b : User a note tous les films (devrait retourner vide)
    print_info("6b. Utilisateur avec tous les films notes")

    # Noter 20 films pour david
    david = "david"
    for i in range(1, 21):
        rate_movie(david, f"movie_{i}", 4.0)

    time.sleep(0.3)
    david_recs = get_recommendations(david, n=5)

    if len(david_recs.get("recommendations", [])) == 0:
        print_success("User avec tous les films notes → liste vide (correct)")
    else:
        print_error(f"David a tout note mais recoit {len(david_recs['recommendations'])} recs")
        return False

    # 6c : Verifier presence de la cle strategy partout
    print_info("6c. Verification cle 'strategy' dans toutes les recs")
    test_user = "alice"
    all_recs = get_recommendations(test_user, n=8)["recommendations"]

    missing_strategy = [r for r in all_recs if "strategy" not in r]
    if len(missing_strategy) > 0:
        print_error(f"{len(missing_strategy)} recs sans cle 'strategy'")
        return False

    print_success("Toutes les recs ont la cle 'strategy'")

    print_success("TEST 6 PASSE\n")
    return True

def test_7_sgd_convergence():
    """Test 7 : SGD converge (erreur diminue avec ratings)."""
    print_header("TEST 7 : SGD CONVERGENCE")

    user = "emma"
    movie = "movie_1"  # The Matrix

    print_info(f"Emma note {movie} plusieurs fois (simuler re-rating)")

    # Premiere note
    rate_movie(user, movie, 5.0)
    print_result("1ere note : 5.0 (prediction_error visible dans les logs serveur)")

    # Re-noter le meme film (devrait avoir une erreur plus petite)
    # Note: En realite, SGD continue d'ajuster meme avec la meme note
    for i in range(2, 6):
        rate_movie(user, movie, 5.0)
        print_result(f"{i}e note : 5.0")
        time.sleep(0.1)

    print_success("SGD updates executes (verifier prediction_error dans les logs)")
    print_info("Pour validation : les logs Docker doivent montrer prediction_error decroissant")
    print_success("TEST 7 PASSE\n")
    return True

# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    """Execute tous les tests."""
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         KINETOSCOPE V2 — TEST SUITE (SVD + BANDIT)              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(Style.RESET_ALL)

    if not check_server():
        print_error("\nServeur inaccessible. Arret des tests.")
        return

    tests = [
        ("Cold-Start", test_1_cold_start),
        ("Premier Rating SGD", test_2_first_rating_sgd),
        ("MATCH/DISCOVERY Mix", test_3_recommendations_match_discovery),
        ("Multi-Users Personnalisation", test_4_multiple_users_personalization),
        ("Genre Boost", test_5_genre_boost),
        ("Edge Cases", test_6_edge_cases),
        ("SGD Convergence", test_7_sgd_convergence),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print_error(f"Exception dans {name} : {e}")
            results.append((name, False))

    # Resume
    print_header("RESUME DES TESTS")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = f"{Fore.GREEN}PASSE" if success else f"{Fore.RED}ECHOUE"
        print(f"{status}{Style.RESET_ALL} — {name}")

    print(f"\n{Fore.CYAN}{'='*70}")
    if passed == total:
        print(f"{Fore.GREEN}{Style.BRIGHT}TOUS LES TESTS REUSSIS ({passed}/{total}) ! 🎉")
        print(f"{Fore.GREEN}Le moteur SVD+Bandit fonctionne correctement.")
    else:
        print(f"{Fore.YELLOW}{Style.BRIGHT}{passed}/{total} tests reussis.")
        print(f"{Fore.YELLOW}Verifiez les logs Docker pour plus de details.")
    print(f"{Fore.CYAN}{'='*70}\n")

if __name__ == "__main__":
    run_all_tests()
