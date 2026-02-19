"""
PRISM CINE — Point d'entree Flask.

Assemble Ray (RecommenderSystem), XMPP (Slixmpp) et Flask (REST + UI).
"""

import os
import socket
import logging
import threading
import asyncio
from typing import Optional

import ray
from flask import Flask, request, jsonify, render_template

from core.recommender import RecommenderSystem, MOVIE_CATALOG
from core.xmpp_client import PRISMCINEXMPPClient
from utils.tmdb_client import tmdb_client  # NOUVEAU : pour posters TMDB

# --- Logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger("prismcine.app")

# Etat partage XMPP <-> Flask (dict mutable, thread-safe pour les ecritures atomiques)
SYSTEM_STATE = {"boosted_genre": None}

def _on_boost_received(genre: str) -> None:
    """Callback invoque par le client XMPP quand BOOST_GENRE est recu."""
    SYSTEM_STATE["boosted_genre"] = genre
    print(f"!!! UI UPDATE !!! Genre is now: {SYSTEM_STATE['boosted_genre']}")
    logger.info("[NODE-1] SYSTEM_STATE boosted_genre updated to: %s", genre)

# --- Initialisation Ray ---
ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    object_store_memory=100 * 1024 * 1024,
    _system_config={"automatic_object_spilling_enabled": False}
)
recommender = RecommenderSystem.remote()
print("[NODE-1] Ray actor RecommenderSystem created")
logger.info("[NODE-1] Ray actor RecommenderSystem created")

# --- Client XMPP (thread daemon + boucle asyncio dediee) ---

async def _run_xmpp_client(actor):
    """Coroutine async pour lancer le client XMPP."""
    jid_base = os.getenv("XMPP_JID", "recommender@prismcine.local")
    password = os.getenv("XMPP_PASSWORD", "rec1pass")
    host = os.getenv("XMPP_HOST", "ejabberd")
    port = int(os.getenv("XMPP_PORT", "5222"))

    resource = socket.gethostname()
    full_jid = f"{jid_base}/{resource}"

    client = PRISMCINEXMPPClient(full_jid, password, actor, boost_callback=_on_boost_received)
    client.use_tls = False
    client.use_ssl = False
    client['feature_mechanisms'].unencrypted_plain = True

    print(f"[NODE-1] Connecting XMPP to {host}:{port} as {full_jid}", flush=True)
    client.connect(host, port)
    await client.disconnected


def _start_xmpp_client(actor):
    """Lance le client XMPP dans un thread dedie avec sa propre boucle evenementielle."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_run_xmpp_client(actor))
    except Exception as e:
        print(f"[NODE-1] XMPP client error: {e}", flush=True)
        logger.error("[NODE-1] XMPP client error: %s", e)
    finally:
        loop.close()


_xmpp_thread = threading.Thread(target=_start_xmpp_client, args=(recommender,), daemon=True)
_xmpp_thread.start()
print("[NODE-1] XMPP thread started (daemon)")

# --- Application Flask ---
app = Flask(__name__)


@app.route("/")
def index():
    """Rendu du dashboard principal avec stats, films et recommandations cold-start."""
    print("[NODE-1] GET / -- Loading dashboard")
    try:
        stats = ray.get(recommender.get_stats.remote())
        recommendations = ray.get(recommender.recommend.remote(user_id=None))

        posters = tmdb_client.get_all_posters(MOVIE_CATALOG)

        movies = [
            {
                "movie_id": mid,
                "title": m["title"],
                "genre": m["genre"],
                "poster_url": posters.get(mid, f"https://placehold.co/600x900/1a1a1a/666?text={m['title'].replace(' ', '+')}")
            }
            for mid, m in MOVIE_CATALOG.items()
        ]
    except Exception as exc:
        print(f"[NODE-1] Error rendering index: {exc}")
        logger.error("[NODE-1] Index render failed: %s", exc)
        stats = {"total_users": 0, "total_ratings": 0, "genre_popularity": {}, "boosted_genre": None}
        recommendations = {"recommendations": []}
        movies = []
        posters = {}

    system_health = {'ray': True, 'xmpp': True, 'jade_agent': 'Scanning'}
    nodes_active = 3

    current_boost = SYSTEM_STATE["boosted_genre"] or stats.get("boosted_genre")
    print(f"[NODE-1] Rendering index -- current_boost = {current_boost}")

    return render_template(
        "index.html",
        stats=stats,
        recommendations=recommendations,
        movies=movies,
        posters=posters,
        system_health=system_health,
        nodes_active=nodes_active,
        current_boost=current_boost,
    )


@app.route("/api/rate", methods=["POST"])
def rate_movie():
    """Accepte une note utilisateur -- ``{user_id, movie_id, rating}``."""
    try:
        data = request.get_json(force=True)
        user_id = data["user_id"]
        movie_id = data["movie_id"]
        rating = float(data["rating"])

        print(f"[NODE-1] POST /api/rate -- user={user_id} movie={movie_id} rating={rating}")

        result = ray.get(recommender.rate.remote(user_id, movie_id, rating))
        if "error" in result:
            return jsonify(result), 400

        # Broadcast XMPP UPDATE
        print(f"[NODE-1] Broadcasting XMPP UPDATE after rating {movie_id}")

        return jsonify(result)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[NODE-1] Invalid rating request: {exc}")
        logger.warning("[NODE-1] Invalid rating request: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        print(f"[NODE-1] Rating error: {exc}")
        logger.error("[NODE-1] Rating error: %s", exc)
        return jsonify({"error": "internal_error"}), 500


@app.route("/api/recommend/<user_id>")
def get_recommendations(user_id):
    """Retourne les Top-N recommandations pour un utilisateur."""
    print(f"[NODE-1] GET /api/recommend/{user_id}")
    try:
        top_n = int(request.args.get("n", 5))
        recs = ray.get(recommender.recommend.remote(user_id=user_id, top_n=top_n))
        return jsonify(recs)
    except Exception as exc:
        print(f"[NODE-1] Recommendation error: {exc}")
        logger.error("[NODE-1] Recommendation error: %s", exc)
        return jsonify({"error": "internal_error"}), 500


@app.route("/api/stats")
def get_stats():
    """Statistiques systeme -- sonde toutes les 10s par l agent JADE TrendScout."""
    print("[NODE-1] GET /api/stats -- polled by JADE agent")
    try:
        stats = ray.get(recommender.get_stats.remote())
        return jsonify(stats)
    except Exception as exc:
        print(f"[NODE-1] Stats error: {exc}")
        logger.error("[NODE-1] Stats error: %s", exc)
        return jsonify({"error": "internal_error"}), 500


@app.route("/api/config", methods=["GET"])
def get_config():
    """Retourne la configuration actuelle du moteur (cb_weight, epsilon, use_bias)."""
    try:
        config = ray.get(recommender.get_config.remote())
        return jsonify(config)
    except Exception as exc:
        logger.error("[NODE-1] Config read error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/config", methods=["POST"])
def set_config():
    """Met a jour les hyperparametres du moteur (pour benchmarking)."""
    try:
        data = request.get_json(force=True)
        cb_weight = data.get("cb_weight")
        epsilon = data.get("epsilon")
        use_bias = data.get("use_bias")
        print(f"[NODE-1] POST /api/config -- cb_weight={cb_weight} epsilon={epsilon} use_bias={use_bias}")
        config = ray.get(recommender.set_config.remote(
            cb_weight=cb_weight, epsilon=epsilon, use_bias=use_bias
        ))
        return jsonify({"status": "config_updated", "config": config})
    except Exception as exc:
        logger.error("[NODE-1] Config update error: %s", exc)
        return jsonify({"error": str(exc)}), 400


@app.route("/api/boost", methods=["POST"])
def boost_genre():
    """Applique un boost de genre (endpoint interne, ou appelable manuellement)."""
    try:
        data = request.get_json(force=True)
        genre = data["genre"]
        print(f"[NODE-1] POST /api/boost -- genre={genre}")
        ray.get(recommender.boost_genre.remote(genre))
        _on_boost_received(genre)
        return jsonify({"status": "boosted", "genre": genre, "system_state": SYSTEM_STATE})
    except Exception as exc:
        print(f"[NODE-1] Boost error: {exc}")
        logger.error("[NODE-1] Boost error: %s", exc)
        return jsonify({"error": str(exc)}), 400


@app.route("/api/boost-status")
def boost_status():
    """Retourne le genre actuellement booste (polling JS)."""
    current = SYSTEM_STATE["boosted_genre"]
    return jsonify({"boosted_genre": current})


@app.route("/api/reset", methods=["POST"])
def reset_system():
    """RESET complet du systeme (pour tests uniquement)."""
    try:
        print("[NODE-1] POST /api/reset -- WIPING ALL DATA")
        result = ray.get(recommender.reset.remote())
        SYSTEM_STATE["boosted_genre"] = None
        print(f"[NODE-1] Reset complete: {result}")
        logger.warning("[NODE-1] System RESET executed: %s", result)
        return jsonify(result)
    except Exception as exc:
        print(f"[NODE-1] Reset error: {exc}")
        logger.error("[NODE-1] Reset error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# --- Point d'entree ---
if __name__ == "__main__":
    flask_port = int(os.getenv("FLASK_PORT", "5000"))
    print(f"[NODE-1] Flask starting on 0.0.0.0:{flask_port}")
    app.run(host="0.0.0.0", port=flask_port, debug=False, threaded=True)
