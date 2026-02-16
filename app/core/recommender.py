"""
Kinetoscope V2 — Moteur de recommandation hybride SVD + Bandit Epsilon-Greedy.

Architecture :
    Factorisation matricielle (SVD) avec apprentissage en ligne (SGD)
    pour capturer les facteurs latents des gouts utilisateurs,
    combinee a une strategie Multi-Armed Bandit (Epsilon-Greedy)
    pour equilibrer Exploitation (films predits comme pertinents)
    et Exploration (decouverte de preferences inconnues).

Ici on utilise Ray car :
    Les Ray actors fournissent des objets *stateful* et *thread-safe*.
    Plusieurs threads Flask peuvent appeler rate() et recommend()
    simultanement ; Ray serialise ces appels par acteur, eliminant
    les race-conditions sur les matrices NumPy partagees sans verrous
    explicites.

Dataset : "MovieLens-Lite" — 20 films hardcodes couvrant 8 genres
    pour simuler un vrai catalogue sans dependance a un fichier externe.
"""

import ray
import random
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger("kinetoscope.recommender")

# ============================================================================
# Hyperparametres SVD + Bandit
#
#   k              : dimension de l espace latent (rang de la factorisation)
#   LEARNING_RATE  : pas de descente SGD (alpha dans les formules)
#   REGULARIZATION : coefficient L2 (lambda) pour eviter le surapprentissage
#   EPSILON        : probabilite d exploration du bandit epsilon-greedy
# ============================================================================
LATENT_FACTORS_K = 10
LEARNING_RATE = 0.05
REGULARIZATION = 0.02
EPSILON = 0.2
GLOBAL_MEAN = 3.0   # Centre de l echelle 1-5 (biais global SVD)
SGD_EPOCHS = 10      # Nombre d iterations SGD par appel a rate()
CB_WEIGHT = 0.7      # Poids du content-based dans le score hybride (0..1)

# ============================================================================
# MovieLens-Lite — 50 films hardcodes, 10 genres equilibres
# En production, ce catalogue serait alimente par une base de donnees.
# tmdb_id : Identifiant TMDB pour recuperer les posters via TMDB API
# ============================================================================
MOVIE_CATALOG: Dict[str, Dict[str, Any]] = {
    # ===== ACTION (5 films) =====
    "movie_1":  {"title": "The Matrix",              "genre": "Action", "tmdb_id": 603},
    "movie_2":  {"title": "Mad Max: Fury Road",      "genre": "Action", "tmdb_id": 76341},
    "movie_3":  {"title": "John Wick",               "genre": "Action", "tmdb_id": 245891},
    "movie_4":  {"title": "Die Hard",                "genre": "Action", "tmdb_id": 562},
    "movie_5":  {"title": "The Dark Knight",         "genre": "Action", "tmdb_id": 155},

    # ===== SCI-FI (5 films) =====
    "movie_6":  {"title": "Interstellar",            "genre": "Sci-Fi", "tmdb_id": 157336},
    "movie_7":  {"title": "Blade Runner 2049",       "genre": "Sci-Fi", "tmdb_id": 335984},
    "movie_8":  {"title": "Arrival",                 "genre": "Sci-Fi", "tmdb_id": 329865},
    "movie_9":  {"title": "Inception",               "genre": "Sci-Fi", "tmdb_id": 27205},
    "movie_10": {"title": "Dune",                    "genre": "Sci-Fi", "tmdb_id": 438631},

    # ===== DRAMA (5 films) =====
    "movie_11": {"title": "The Shawshank Redemption", "genre": "Drama", "tmdb_id": 278},
    "movie_12": {"title": "Forrest Gump",            "genre": "Drama", "tmdb_id": 13},
    "movie_13": {"title": "The Godfather",           "genre": "Drama", "tmdb_id": 238},
    "movie_14": {"title": "Schindler's List",        "genre": "Drama", "tmdb_id": 424},
    "movie_15": {"title": "Fight Club",              "genre": "Drama", "tmdb_id": 550},

    # ===== ROMANCE (5 films) =====
    "movie_16": {"title": "The Notebook",            "genre": "Romance", "tmdb_id": 11036},
    "movie_17": {"title": "Titanic",                 "genre": "Romance", "tmdb_id": 597},
    "movie_18": {"title": "La La Land",              "genre": "Romance", "tmdb_id": 313369},
    "movie_19": {"title": "Pride and Prejudice",     "genre": "Romance", "tmdb_id": 17647},
    "movie_20": {"title": "Eternal Sunshine",        "genre": "Romance", "tmdb_id": 38},

    # ===== ANIMATION (5 films) =====
    "movie_21": {"title": "Spirited Away",           "genre": "Animation", "tmdb_id": 129},
    "movie_22": {"title": "Spider-Verse",            "genre": "Animation", "tmdb_id": 324857},
    "movie_23": {"title": "Toy Story",               "genre": "Animation", "tmdb_id": 862},
    "movie_24": {"title": "Up",                      "genre": "Animation", "tmdb_id": 14160},
    "movie_25": {"title": "WALL-E",                  "genre": "Animation", "tmdb_id": 10681},

    # ===== THRILLER (5 films) =====
    "movie_26": {"title": "Se7en",                   "genre": "Thriller", "tmdb_id": 807},
    "movie_27": {"title": "Gone Girl",               "genre": "Thriller", "tmdb_id": 210577},
    "movie_28": {"title": "Shutter Island",          "genre": "Thriller", "tmdb_id": 11324},
    "movie_29": {"title": "Prisoners",               "genre": "Thriller", "tmdb_id": 146233},
    "movie_30": {"title": "Zodiac",                  "genre": "Thriller", "tmdb_id": 1949},

    # ===== HORROR (5 films) =====
    "movie_31": {"title": "Get Out",                 "genre": "Horror", "tmdb_id": 419430},
    "movie_32": {"title": "The Shining",             "genre": "Horror", "tmdb_id": 694},
    "movie_33": {"title": "Hereditary",              "genre": "Horror", "tmdb_id": 493922},
    "movie_34": {"title": "A Quiet Place",           "genre": "Horror", "tmdb_id": 447332},
    "movie_35": {"title": "The Conjuring",           "genre": "Horror", "tmdb_id": 138843},

    # ===== COMEDY (5 films) =====
    "movie_36": {"title": "The Grand Budapest Hotel", "genre": "Comedy", "tmdb_id": 120467},
    "movie_37": {"title": "Superbad",                "genre": "Comedy", "tmdb_id": 8363},
    "movie_38": {"title": "The Big Lebowski",        "genre": "Comedy", "tmdb_id": 115},
    "movie_39": {"title": "Knives Out",              "genre": "Comedy", "tmdb_id": 546554},
    "movie_40": {"title": "Parasite",                "genre": "Comedy", "tmdb_id": 496243},

    # ===== FANTASY (5 films) =====
    "movie_41": {"title": "The Lord of the Rings",   "genre": "Fantasy", "tmdb_id": 120},
    "movie_42": {"title": "Harry Potter",            "genre": "Fantasy", "tmdb_id": 671},
    "movie_43": {"title": "Pan's Labyrinth",         "genre": "Fantasy", "tmdb_id": 1121},
    "movie_44": {"title": "The Shape of Water",      "genre": "Fantasy", "tmdb_id": 399055},
    "movie_45": {"title": "Stardust",                "genre": "Fantasy", "tmdb_id": 1733},

    # ===== ADVENTURE (5 films) =====
    "movie_46": {"title": "Indiana Jones",           "genre": "Adventure", "tmdb_id": 85},
    "movie_47": {"title": "Jurassic Park",           "genre": "Adventure", "tmdb_id": 329},
    "movie_48": {"title": "The Revenant",            "genre": "Adventure", "tmdb_id": 281957},
    "movie_49": {"title": "Mad Max 2",               "genre": "Adventure", "tmdb_id": 9659},
    "movie_50": {"title": "Life of Pi",              "genre": "Adventure", "tmdb_id": 87827},
}


@ray.remote
class RecommenderSystem:
    """
    Stateful Ray actor implementing a hybrid recommendation engine :

        1. Factorisation Matricielle (SVD)
           R ≈ P . Q^T
           ou P (n_users x k) et Q (n_items x k) sont les matrices
           de facteurs latents, apprises par SGD a chaque vote.

        2. Multi-Armed Bandit (Epsilon-Greedy)
           - Exploitation (1 - epsilon) : films avec le plus haut P_u . Q_i
           - Exploration  (epsilon)     : film aleatoire pour decouvrir
             de nouvelles preferences utilisateur.

    Ici on utilise un Ray Actor car les matrices sont partagees entre toutes
    les requetes Flask -- Ray garantit la coherence sans lock explicite.
    """

    def __init__(self):
        self.movie_ids: List[str] = list(MOVIE_CATALOG.keys())
        self.movie_index: Dict[str, int] = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.user_ids: List[str] = []
        self.user_index: Dict[str, int] = {}

        # Matrice 2-D : lignes = utilisateurs, colonnes = films. 0.0 = non note.
        # Conservee pour savoir quels films ont ete notes (masque d exclusion).
        self.ratings: np.ndarray = np.zeros((0, len(self.movie_ids)), dtype=np.float64)

        # -- Factorisation matricielle SVD avec biais : R ≈ mu + b_u + b_i + P . Q^T --
        # mu : biais global (moyenne de l echelle 1-5 = 3.0)
        # b_u : biais utilisateur (tend-il a noter haut ou bas ?)
        # b_i : biais item (ce film est-il generalement bien note ?)
        # P : matrice des facteurs latents utilisateurs (n_users x k)
        # Q : matrice des facteurs latents items       (n_items x k)
        #
        # Initialises avec uniform(0.3, 0.7) pour briser la symetrie.
        self.latent_dim: int = LATENT_FACTORS_K
        self.learning_rate: float = LEARNING_RATE
        self.regularization: float = REGULARIZATION
        self.epsilon: float = EPSILON
        self.global_mean: float = GLOBAL_MEAN

        # Q est fixe en taille (50 films) -- on l initialise immediatement.
        self.item_latent_factors: np.ndarray = np.random.uniform(
            -0.01, 0.01, size=(len(self.movie_ids), self.latent_dim)
        )

        # P demarre vide et grandit avec chaque nouvel utilisateur.
        self.user_latent_factors: np.ndarray = np.zeros(
            (0, self.latent_dim), dtype=np.float64
        )

        # Biais item : un scalaire par film (initialise a 0)
        self.item_bias: np.ndarray = np.zeros(len(self.movie_ids), dtype=np.float64)
        # Biais utilisateur : grandit dynamiquement avec les users
        self.user_bias: np.ndarray = np.zeros(0, dtype=np.float64)

        # Etat de boost genre -- defini par les messages XMPP BOOST_GENRE.
        self.boosted_genre: Optional[str] = None
        self.boost_factor: float = 0.2

        # Compteurs pour /api/stats consomme par l agent JADE.
        self.total_ratings: int = 0
        self.rating_counts_by_genre: Dict[str, int] = {
            m["genre"]: 0 for m in MOVIE_CATALOG.values()
        }
        self.start_time: float = time.time()

        num_genres = len(set(m["genre"] for m in MOVIE_CATALOG.values()))
        print(f"[RAY-ACTOR] RecommenderSystem initialised (SVD k={self.latent_dim}, "
              f"lr={self.learning_rate}, reg={self.regularization}, "
              f"epsilon={self.epsilon}) | {len(self.movie_ids)} movies, {num_genres} genres")
        logger.info("[RAY-ACTOR] RecommenderSystem actor initialised with %d movies (SVD mode)",
                    len(self.movie_ids))

    # ------------------------------------------------------------------
    # Helpers prives
    # ------------------------------------------------------------------

    def _ensure_user(self, user_id: str) -> int:
        """
        Enregistre un nouvel utilisateur dans la matrice si premiere visite.

        Cold-start SVD : le vecteur latent est initialise avec du bruit
        faible et le biais utilisateur a 0 (neutre). La prediction initiale
        sera mu + 0 + b_i + ~0 = ~3.0 (centre de l echelle).
        """
        if user_id not in self.user_index:
            idx = len(self.user_ids)
            self.user_ids.append(user_id)
            self.user_index[user_id] = idx

            # Extension de la matrice des notes
            new_row = np.zeros((1, len(self.movie_ids)), dtype=np.float64)
            self.ratings = (np.vstack([self.ratings, new_row])
                            if self.ratings.size > 0 else new_row)

            # Cold-start : vecteur latent avec bruit faible
            new_latent_vector = np.random.uniform(
                -0.01, 0.01, size=(1, self.latent_dim)
            )
            self.user_latent_factors = (
                np.vstack([self.user_latent_factors, new_latent_vector])
                if self.user_latent_factors.size > 0
                else new_latent_vector
            )

            # Biais utilisateur initialise a 0 (neutre)
            self.user_bias = np.append(self.user_bias, 0.0)

            print(f"[RAY-ACTOR] New user registered: {user_id} (index={idx})")
        return self.user_index[user_id]

    def _sgd_update(self, user_idx: int, movie_idx: int, actual_rating: float,
                    update_items: bool = True) -> float:
        """
        Mise a jour SGD (Stochastic Gradient Descent) des facteurs latents
        et des biais.

        Formulation mathematique (SVD avec biais) :
            prediction  = mu + b_u + b_i + P_u . Q_i^T
            erreur      = r_ui - prediction

            # Mise a jour des biais :
            b_u <- b_u + alpha * (erreur - lambda * b_u)
            b_i <- b_i + alpha * (erreur - lambda * b_i)   [si update_items]

            # Mise a jour des facteurs latents :
            P_u <- P_u + alpha * (erreur * Q_i - lambda * P_u)
            Q_i <- Q_i + alpha * (erreur * P_u - lambda * Q_i)   [si update_items]

        Args:
            update_items : si False, seuls P_u et b_u sont mis a jour.
                Cela evite la cross-contamination des facteurs items
                partages entre utilisateurs lors du multi-epoch.

        Returns:
            prediction_error : l erreur avant mise a jour (monitoring).
        """
        # Copie des vecteurs actuels (important : utiliser les valeurs
        # AVANT mise a jour pour les deux gradients)
        user_latent_vector = self.user_latent_factors[user_idx].copy()
        item_latent_vector = self.item_latent_factors[movie_idx].copy()

        # Prediction SVD avec biais : mu + b_u + b_i + P_u . Q_i
        predicted_rating = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias[movie_idx]
            + np.dot(user_latent_vector, item_latent_vector)
        )

        # Erreur de prediction
        prediction_error = actual_rating - predicted_rating

        # Mise a jour biais utilisateur (toujours)
        self.user_bias[user_idx] += self.learning_rate * (
            prediction_error - self.regularization * self.user_bias[user_idx]
        )

        # Mise a jour biais item (seulement si update_items)
        if update_items:
            self.item_bias[movie_idx] += self.learning_rate * (
                prediction_error - self.regularization * self.item_bias[movie_idx]
            )

        # SGD Update Rule: P_u = P_u + lr * (err * Q_i - reg * P_u)
        self.user_latent_factors[user_idx] += self.learning_rate * (
            prediction_error * item_latent_vector
            - self.regularization * user_latent_vector
        )

        # SGD Update Rule: Q_i (seulement si update_items)
        if update_items:
            self.item_latent_factors[movie_idx] += self.learning_rate * (
                prediction_error * user_latent_vector
                - self.regularization * item_latent_vector
            )

        return prediction_error

    def _compute_genre_affinity(self, user_idx: int) -> np.ndarray:
        """
        Content-Based Filtering : calcule un score d affinite par film
        base sur la note moyenne de l utilisateur pour chaque genre.

        Pour chaque film non note, le score est la moyenne des notes
        que l utilisateur a donnees aux films du meme genre.
        Si l utilisateur n a note aucun film de ce genre, on utilise
        la moyenne globale (GLOBAL_MEAN = 3.0).

        Cela fournit un signal fort et immediat base sur les features
        du contenu (genre), contrairement au SVD qui necessite beaucoup
        de donnees pour converger.
        """
        user_ratings = self.ratings[user_idx]

        # Calculer la moyenne par genre pour cet utilisateur
        genre_sums: Dict[str, float] = {}
        genre_counts: Dict[str, int] = {}
        for midx in range(len(self.movie_ids)):
            if user_ratings[midx] > 0:
                genre = MOVIE_CATALOG[self.movie_ids[midx]]["genre"]
                genre_sums[genre] = genre_sums.get(genre, 0.0) + user_ratings[midx]
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        genre_avg: Dict[str, float] = {}
        for g in genre_sums:
            genre_avg[g] = genre_sums[g] / genre_counts[g]

        # Pour chaque film, le score content-based = moyenne du genre
        cb_scores = np.full(len(self.movie_ids), self.global_mean, dtype=np.float64)
        for midx, mid in enumerate(self.movie_ids):
            genre = MOVIE_CATALOG[mid]["genre"]
            if genre in genre_avg:
                cb_scores[midx] = genre_avg[genre]

        return cb_scores

    def _apply_genre_boost(self, scores: np.ndarray) -> np.ndarray:
        """Boost multiplicatif pour les films du genre tendance."""
        if self.boosted_genre is None:
            return scores
        boosted = scores.copy()
        for i, mid in enumerate(self.movie_ids):
            if MOVIE_CATALOG[mid]["genre"] == self.boosted_genre:
                boosted[i] *= (1.0 + self.boost_factor)
        return boosted

    def _cold_start_recommendations(self, top_n: int, exclude_user: Optional[str] = None) -> Dict:
        """
        Fallback quand il n y a pas de signal SVD :
        - Zero note globalement -> retourne l ordre du catalogue (DISCOVERY).
        - Sinon -> classement par moyenne de colonne (MATCH).
        """
        if self.total_ratings == 0:
            movies = [
                {
                    "movie_id": mid,
                    "title": m["title"],
                    "genre": m["genre"],
                    "predicted_score": 0.0,
                    "strategy": "DISCOVERY",
                }
                for mid, m in list(MOVIE_CATALOG.items())[:top_n]
            ]
            return {"user_id": None, "recommendations": movies, "note": "cold_start_no_data"}

        rated_mask = self.ratings > 0
        col_sums = np.sum(self.ratings, axis=0)
        col_counts = np.sum(rated_mask, axis=0).astype(np.float64)
        avg_ratings = np.divide(
            col_sums, col_counts,
            out=np.zeros_like(col_sums),
            where=col_counts > 0,
        )

        avg_ratings = self._apply_genre_boost(avg_ratings)
        top_indices = np.argsort(avg_ratings)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            if avg_ratings[idx] > 0:
                mid = self.movie_ids[idx]
                recommendations.append({
                    "movie_id": mid,
                    "title": MOVIE_CATALOG[mid]["title"],
                    "genre": MOVIE_CATALOG[mid]["genre"],
                    "predicted_score": round(float(avg_ratings[idx]), 2),
                    "strategy": "MATCH",
                })
        return {"user_id": None, "recommendations": recommendations, "note": "cold_start"}

    # ------------------------------------------------------------------
    # API Publique (appelee via ray.get)
    # ------------------------------------------------------------------

    def rate(self, user_id: str, movie_id: str, rating: float) -> Dict:
        """
        Enregistre une note utilisateur (1.0-5.0) et met a jour
        les facteurs latents par descente de gradient stochastique (SGD).

        Online Learning multi-epoch : a chaque nouveau vote, on re-itere
        SGD_EPOCHS fois sur TOUS les ratings existants de cet utilisateur.
        Cela permet aux facteurs latents de converger rapidement meme
        avec peu de donnees (cold-start).
        """
        if movie_id not in self.movie_index:
            return {"error": f"Unknown movie: {movie_id}"}
        if not (1.0 <= rating <= 5.0):
            return {"error": "Rating must be between 1.0 and 5.0"}

        user_idx = self._ensure_user(user_id)
        movie_idx = self.movie_index[movie_id]

        # Stocker la note dans la matrice (sert a savoir quels films
        # ont ete notes pour l exclusion dans recommend())
        self.ratings[user_idx, movie_idx] = rating

        # -- Multi-epoch SGD pour affiner le profil utilisateur --
        # Epoch 0 : mise a jour complete (P_u, Q_i, b_u, b_i)
        # Epochs 1..N : mise a jour user-only (P_u, b_u) pour eviter
        # la cross-contamination des facteurs items partages.
        rated_indices = np.where(self.ratings[user_idx] > 0)[0]
        final_error = 0.0
        for midx in rated_indices:
            stored_rating = self.ratings[user_idx, midx]
            final_error = self._sgd_update(user_idx, midx, stored_rating, update_items=True)
        for _epoch in range(SGD_EPOCHS - 1):
            for midx in rated_indices:
                stored_rating = self.ratings[user_idx, midx]
                final_error = self._sgd_update(user_idx, midx, stored_rating, update_items=False)

        # Compteurs pour /api/stats consomme par l agent JADE
        genre = MOVIE_CATALOG[movie_id]["genre"]
        self.rating_counts_by_genre[genre] = self.rating_counts_by_genre.get(genre, 0) + 1
        self.total_ratings += 1

        title = MOVIE_CATALOG[movie_id]["title"]
        print(f"[RAY-ACTOR] SVD updated: user={user_id} -> {title} ({genre}) = {rating:.1f}"
              f"  |  SGD {SGD_EPOCHS} epochs x {len(rated_indices)} ratings"
              f"  |  final_error={final_error:.4f}"
              f"  |  Total votes: {self.total_ratings}")
        logger.info("[RAY-ACTOR] Rating recorded (SVD): user=%s movie=%s rating=%.1f",
                    user_id, movie_id, rating)
        return {"status": "ok", "user_id": user_id, "movie_id": movie_id, "rating": rating}

    def recommend(self, user_id: Optional[str] = None, top_n: int = 5) -> Dict:
        """
        Recommandation hybride : SVD + Content-Based + Bandit Epsilon-Greedy.

        Pipeline :
            1. SVD score  : mu + b_u + b_i + P_u . Q_i  (collaborative filtering)
            2. CB score   : moyenne des notes user par genre  (content-based)
            3. Hybrid     : (1 - CB_WEIGHT) * svd + CB_WEIGHT * cb
            4. Clamper dans [0, 5]
            5. Appliquer le boost de genre (si actif via JADE/XMPP)
            6. Masquer les films deja notes
            7. Strategie epsilon-greedy :
               - Avec probabilite (1 - epsilon) : Top-N par score (MATCH)
               - Avec probabilite epsilon       : film aleatoire non note (DISCOVERY)

        Le hybride garantit que les recommendations refletent les preferences
        de genre meme avec peu de donnees (cold-start), tandis que le SVD
        affine la personnalisation au fur et a mesure.
        """
        if user_id is None or user_id not in self.user_index:
            print(f"[RAY-ACTOR] Cold-start recommendations requested (user={user_id})")
            return self._cold_start_recommendations(top_n)

        user_idx = self.user_index[user_id]
        user_vector = self.ratings[user_idx]

        # Cas limite : tous les films ont ete notes
        unrated_mask = user_vector == 0.0
        if not np.any(unrated_mask):
            return {"user_id": user_id, "recommendations": [], "note": "all_rated"}

        # -- Score SVD avec biais (Collaborative Filtering) --
        svd_scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.user_latent_factors[user_idx] @ self.item_latent_factors.T
        )
        # Clamper SVD AVANT le blending pour eviter que des biais items
        # gonfles (b_i >> 0 si genre populaire) ne dominent le CB.
        svd_scores = np.clip(svd_scores, 0.0, 5.0)

        # -- Score Content-Based (genre affinity) --
        cb_scores = self._compute_genre_affinity(user_idx)

        # -- Score Hybride : ponderation SVD + Content-Based --
        predicted_scores = (1 - CB_WEIGHT) * svd_scores + CB_WEIGHT * cb_scores

        # Score final dans [0, 5]
        predicted_scores = np.clip(predicted_scores, 0.0, 5.0)

        # Appliquer le boost de genre (multiplicatif, declenche par JADE)
        predicted_scores = self._apply_genre_boost(predicted_scores)

        # Masquer les films deja notes (score -> -inf pour le tri)
        predicted_scores[~unrated_mask] = -np.inf

        # -- Bandit Epsilon-Greedy : compromis Exploration vs Exploitation --
        # Indices des films non notes, tries par score decroissant
        unrated_indices = np.where(unrated_mask)[0]
        sorted_unrated = unrated_indices[
            np.argsort(predicted_scores[unrated_indices])[::-1]
        ]

        # Nombre de slots exploitation vs exploration
        # Au moins 1 slot DISCOVERY pour garantir l exploration
        n_explore = max(1, int(top_n * self.epsilon))
        n_exploit = top_n - n_explore

        # Clamper si pas assez de films non notes disponibles
        n_exploit = min(n_exploit, len(sorted_unrated))
        n_explore = min(n_explore, max(0, len(sorted_unrated) - n_exploit))

        recommendations = []

        # -- Exploitation : les films avec le plus haut P_u . Q_i (MATCH) --
        exploit_indices = sorted_unrated[:n_exploit]
        for idx in exploit_indices:
            mid = self.movie_ids[idx]
            score = predicted_scores[idx]
            recommendations.append({
                "movie_id": mid,
                "title": MOVIE_CATALOG[mid]["title"],
                "genre": MOVIE_CATALOG[mid]["genre"],
                "predicted_score": round(float(score), 2),
                "strategy": "MATCH",
            })

        # -- Exploration : film aleatoire parmi les non-selectionnes (DISCOVERY) --
        # Le bandit "tire un bras" au hasard pour decouvrir de nouvelles preferences
        remaining_indices = sorted_unrated[n_exploit:]
        if len(remaining_indices) > 0 and n_explore > 0:
            explore_picks = np.random.choice(
                remaining_indices,
                size=min(n_explore, len(remaining_indices)),
                replace=False,
            )
            for idx in explore_picks:
                mid = self.movie_ids[idx]
                score = predicted_scores[idx]
                recommendations.append({
                    "movie_id": mid,
                    "title": MOVIE_CATALOG[mid]["title"],
                    "genre": MOVIE_CATALOG[mid]["genre"],
                    "predicted_score": round(float(score), 2),
                    "strategy": "DISCOVERY",
                })

        match_count = sum(1 for r in recommendations if r["strategy"] == "MATCH")
        discovery_count = sum(1 for r in recommendations if r["strategy"] == "DISCOVERY")
        print(f"[RAY-ACTOR] Recommendations for {user_id}: "
              f"{len(recommendations)} items (SVD+Bandit: "
              f"{match_count} MATCH, {discovery_count} DISCOVERY)")
        logger.info("[RAY-ACTOR] Recommendations for %s: %d items (%d MATCH, %d DISCOVERY)",
                    user_id, len(recommendations), match_count, discovery_count)
        return {"user_id": user_id, "recommendations": recommendations}

    def boost_genre(self, genre: str) -> None:
        """Definit le genre booste (declenche par message XMPP BOOST_GENRE)."""
        print(f"[RAY-ACTOR] Genre boost updated: {self.boosted_genre} -> {genre}")
        logger.info("[RAY-ACTOR] Genre boost updated: %s -> %s", self.boosted_genre, genre)
        self.boosted_genre = genre

    def get_stats(self) -> Dict:
        """
        Statistiques systeme consommees par l agent JADE TrendScout.

        genre_popularity : fractions du total (pour l UI).
        genre_votes      : compteurs absolus (pour le seuil JADE > 5).
        """
        genre_popularity = {}
        for genre, count in self.rating_counts_by_genre.items():
            genre_popularity[genre] = round(count / max(self.total_ratings, 1), 3)

        print(f"[RAY-ACTOR] Stats polled -- {self.total_ratings} ratings, "
              f"{len(self.user_ids)} users, boosted={self.boosted_genre}")

        return {
            "total_users": len(self.user_ids),
            "total_ratings": self.total_ratings,
            "total_movies": len(self.movie_ids),
            "genre_popularity": genre_popularity,
            "genre_votes": dict(self.rating_counts_by_genre),
            "boosted_genre": self.boosted_genre,
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }

    def reset(self) -> Dict:
        """
        Reinitialise completement le systeme (DANGEREUX — pour tests uniquement).

        Remet a zero :
            - Tous les utilisateurs et leurs vecteurs latents P
            - Toutes les notes dans la matrice ratings
            - Les facteurs latents items Q (re-initialises avec bruit gaussien)
            - Les compteurs de stats et le boost genre

        Retourne le nombre d utilisateurs/ratings supprimes.
        """
        old_users = len(self.user_ids)
        old_ratings = self.total_ratings

        # Reset users et matrice de notes
        self.user_ids = []
        self.user_index = {}
        self.ratings = np.zeros((0, len(self.movie_ids)), dtype=np.float64)

        # Reset facteurs latents P (vide) et Q (re-initialise)
        self.user_latent_factors = np.zeros((0, self.latent_dim), dtype=np.float64)
        self.item_latent_factors = np.random.uniform(
            -0.01, 0.01, size=(len(self.movie_ids), self.latent_dim)
        )

        # Reset biais
        self.user_bias = np.zeros(0, dtype=np.float64)
        self.item_bias = np.zeros(len(self.movie_ids), dtype=np.float64)

        # Reset boost et compteurs
        self.boosted_genre = None
        self.total_ratings = 0
        self.rating_counts_by_genre = {m["genre"]: 0 for m in MOVIE_CATALOG.values()}
        self.start_time = time.time()

        print(f"[RAY-ACTOR] RESET executed — deleted {old_users} users, {old_ratings} ratings")
        logger.warning("[RAY-ACTOR] System RESET — all data wiped (users=%d, ratings=%d)",
                       old_users, old_ratings)

        return {
            "status": "reset_complete",
            "deleted_users": old_users,
            "deleted_ratings": old_ratings,
            "message": "All data wiped — SVD matrices re-initialized",
        }
