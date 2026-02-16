#!/usr/bin/env python3
"""
Script helper pour reinitialiser le systeme PRISM CINE.

Usage:
    python reset_system.py
"""

import requests
import sys

BASE_URL = "http://localhost:5000"

def reset_system():
    """Envoie une requete POST /api/reset."""
    print(" Reinitialisation du systeme en cours...")
    print("  Toutes les donnees seront supprimees (users, ratings, facteurs latents SVD)")

    confirm = input("\nConfirmer la reinitialisation ? (oui/non): ").strip().lower()

    if confirm not in ["oui", "yes", "y", "o"]:
        print(" Reinitialisation annulee.")
        return

    try:
        resp = requests.post(f"{BASE_URL}/api/reset", timeout=5)
        resp.raise_for_status()
        result = resp.json()

        print("\n REINITIALISATION REUSSIE")
        print(f"   - Users supprimes: {result['deleted_users']}")
        print(f"   - Ratings supprimes: {result['deleted_ratings']}")
        print(f"   - Message: {result['message']}")
        print("\n Le systeme est maintenant en etat cold-start.")

    except requests.exceptions.ConnectionError:
        print("\n ERREUR : Impossible de contacter le serveur")
        print("   Verifiez que docker-compose up tourne sur le port 5000")
        sys.exit(1)

    except Exception as e:
        print(f"\n ERREUR : {e}")
        sys.exit(1)

if __name__ == "__main__":
    reset_system()
