#!/usr/bin/env python3
"""
Kinetoscope — Script de simulation de trafic pour la soutenance.

Ce script bombarde le serveur Flask de notes sur les films Sci-Fi
pour forcer l agent JADE a detecter la tendance et declencher la
banniere "TREND DETECTED" dans l UI.

Zero dependances externes : utilise uniquement urllib (stdlib).

Usage :
    python3 simulate_traffic.py                        # cible localhost:5000
    python3 simulate_traffic.py http://localhost:32781  # port Docker custom
"""

import sys
import time
import json
import random
import urllib.request
import urllib.error

# -- Configuration -----------------------------------------------------------
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
RATE_ENDPOINT = f"{BASE_URL}/api/rate"

# Cibles Sci-Fi (IDs du recommender.py hardcode)
# SCI_FI_TARGETS = [
#     {"movie_id": "movie_3", "title": "Interstellar"},
#     {"movie_id": "movie_4", "title": "Blade Runner 2049"},
#     {"movie_id": "movie_5", "title": "Arrival"},
# ]

SCI_FI_TARGETS = [
    {"movie_id": "movie_7", "title": "Forrest Gump"},
    {"movie_id": "movie_8", "title": "The Godfather"},
    {"movie_id": "movie_6", "title": "The Shawshank Redemption"},
]
# Utilisateurs fictifs varies pour simuler du trafic realiste
FAKE_USERS = [
    "alice", "bob", "charlie", "diana", "eve",
    "frank", "grace", "heidi", "ivan", "judy",
]

TOTAL_REQUESTS = 20
DELAY_BETWEEN = 0.25  # ~5 secondes pour 20 requetes

# -- ANSI Colors pour les logs terminaux -------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"
TAG = f"{BOLD}{MAGENTA}[SIMULATION]{RESET} "


def post_json(url: str, data: dict, timeout: int = 5) -> dict:
    """POST JSON avec urllib (stdlib) -- pas besoin de requests."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print()
    print(f"{TAG}{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{TAG}{BOLD}{CYAN}  Cine-Trend Traffic Simulator{RESET}")
    print(f"{TAG}{CYAN}  Target  : {RESET}{BASE_URL}")
    print(f"{TAG}{CYAN}  Genre   : {RESET}Sci-Fi (3 films)")
    print(f"{TAG}{CYAN}  Requests: {RESET}{TOTAL_REQUESTS} over ~{TOTAL_REQUESTS * DELAY_BETWEEN:.0f}s")
    print(f"{TAG}{BOLD}{CYAN}{'=' * 60}{RESET}")
    print()

    successes = 0
    failures = 0

    for i in range(1, TOTAL_REQUESTS + 1):
        target = random.choice(SCI_FI_TARGETS)
        user = random.choice(FAKE_USERS)
        rating = round(random.uniform(3.5, 5.0), 1)

        payload = {
            "user_id": user,
            "movie_id": target["movie_id"],
            "rating": rating,
        }

        try:
            print(f"{TAG}{YELLOW}[{i:02d}/{TOTAL_REQUESTS}]{RESET} "
                  f"Voting for {BOLD}{target['title']}{RESET} "
                  f"(user={CYAN}{user}{RESET}, score={GREEN}{rating}{RESET})")

            data = post_json(RATE_ENDPOINT, payload)

            if data.get("status") == "ok":
                print(f"{TAG}  {GREEN}>> Accepted{RESET}")
                successes += 1
            else:
                print(f"{TAG}  {RED}>> Rejected: {data.get('error', 'unknown')}{RESET}")
                failures += 1

        except urllib.error.URLError as e:
            print(f"{TAG}  {RED}>> Connection failed -- is Flask running at {BASE_URL}?{RESET}")
            print(f"{TAG}  {RED}   {e.reason}{RESET}")
            failures += 1
        except Exception as exc:
            print(f"{TAG}  {RED}>> Error: {exc}{RESET}")
            failures += 1

        time.sleep(DELAY_BETWEEN)

    # -- Resume final --------------------------------------------------------
    print()
    print(f"{TAG}{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{TAG}{BOLD}Simulation complete{RESET}")
    print(f"{TAG}  {GREEN}Successes : {successes}{RESET}")
    print(f"{TAG}  {RED}Failures  : {failures}{RESET}")
    print(f"{TAG}{BOLD}{CYAN}{'=' * 60}{RESET}")
    print()

    if successes > 5:
        print(f"{TAG}{BOLD}{GREEN}Sci-Fi threshold (>5 votes) exceeded!{RESET}")
        print(f"{TAG}{YELLOW}   -> JADE agent should detect trend within ~10 seconds{RESET}")
        print(f"{TAG}{YELLOW}   -> Refresh browser to see TREND DETECTED banner{RESET}")
    else:
        print(f"{TAG}{YELLOW}Not enough votes reached. Check Flask connection.{RESET}")

    print()


if __name__ == "__main__":
    main()
