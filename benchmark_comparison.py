#!/usr/bin/env python3
"""
PRISM CINE — Benchmark Comparison Module

Compare le moteur hybride contre des baselines (Pure SVD, Biased SVD,
Content-Based only, Hybrid sans exploration).

Metriques : MAE, GHR (Genre Hit Rate), Genre Coverage, Latence.

Usage:
    python benchmark_comparison.py              # full (5 users × 20 votes)
    python benchmark_comparison.py --quick      # demo (2 users × 10 votes)
    python benchmark_comparison.py --url http://host:port
"""

import argparse
import csv
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' package required.  pip install requests")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not found — plots will be skipped. pip install matplotlib")

# --- Constants ---
SEED = 42
BASE_URL_DEFAULT = "http://localhost:5000"
TIMEOUT = 10  # seconds per HTTP call

# Catalogue miroir (IDs + genres)
MOVIE_CATALOG = {
    "movie_1":  "Action",    "movie_2":  "Action",    "movie_3":  "Action",
    "movie_4":  "Action",    "movie_5":  "Action",
    "movie_6":  "Sci-Fi",    "movie_7":  "Sci-Fi",    "movie_8":  "Sci-Fi",
    "movie_9":  "Sci-Fi",    "movie_10": "Sci-Fi",
    "movie_11": "Drama",     "movie_12": "Drama",     "movie_13": "Drama",
    "movie_14": "Drama",     "movie_15": "Drama",
    "movie_16": "Romance",   "movie_17": "Romance",   "movie_18": "Romance",
    "movie_19": "Romance",   "movie_20": "Romance",
    "movie_21": "Animation", "movie_22": "Animation", "movie_23": "Animation",
    "movie_24": "Animation", "movie_25": "Animation",
    "movie_26": "Thriller",  "movie_27": "Thriller",  "movie_28": "Thriller",
    "movie_29": "Thriller",  "movie_30": "Thriller",
    "movie_31": "Horror",    "movie_32": "Horror",    "movie_33": "Horror",
    "movie_34": "Horror",    "movie_35": "Horror",
    "movie_36": "Comedy",    "movie_37": "Comedy",    "movie_38": "Comedy",
    "movie_39": "Comedy",    "movie_40": "Comedy",
    "movie_41": "Fantasy",   "movie_42": "Fantasy",   "movie_43": "Fantasy",
    "movie_44": "Fantasy",   "movie_45": "Fantasy",
    "movie_46": "Adventure", "movie_47": "Adventure", "movie_48": "Adventure",
    "movie_49": "Adventure", "movie_50": "Adventure",
}

ALL_GENRES = sorted(set(MOVIE_CATALOG.values()))
MOVIES_BY_GENRE: Dict[str, List[str]] = {}
for _mid, _genre in MOVIE_CATALOG.items():
    MOVIES_BY_GENRE.setdefault(_genre, []).append(_mid)

# --- Profils utilisateurs simules ---
USER_PROFILES = {
    "bench_aymen":   {"preferred_genres": ["Sci-Fi", "Thriller"],   "rating_bias": 0.80},
    "bench_ichrak":     {"preferred_genres": ["Action", "Adventure"],  "rating_bias": 0.75},
    "bench_sara":   {"preferred_genres": ["Drama", "Romance"],     "rating_bias": 0.75},
    "bench_mohammed":    {"preferred_genres": ["Comedy", "Animation"],  "rating_bias": 0.70},
    "bench_oualid":     {"preferred_genres": ["Horror", "Fantasy"],    "rating_bias": 0.70},
}

# --- Approches a comparer ---
APPROACHES = [
    {"name": "Pure SVD",            "cb_weight": 0.0,  "epsilon": 0.0, "use_bias": False},
    {"name": "Biased SVD",          "cb_weight": 0.0,  "epsilon": 0.0, "use_bias": True},
    {"name": "Content-Based only",  "cb_weight": 1.0,  "epsilon": 0.0, "use_bias": True},
    {"name": "Hybrid (current)",    "cb_weight": 0.7,  "epsilon": 0.2, "use_bias": True},
    {"name": "Hybrid (no explore)", "cb_weight": 0.7,  "epsilon": 0.0, "use_bias": True},
]

# Alpha sweep
ALPHA_SWEEP_VALUES = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]


def generate_ground_truth(rng: random.Random, profile: Dict) -> Dict[str, float]:
    """Note simulee par film : genres preferes -> 4.0-5.0, autres -> 2.0-3.5."""
    truth: Dict[str, float] = {}
    for mid, genre in MOVIE_CATALOG.items():
        if genre in profile["preferred_genres"]:
            truth[mid] = round(rng.uniform(4.0, 5.0), 1)
        else:
            truth[mid] = round(rng.uniform(2.0, 3.5), 1)
    return truth


RESERVED_PREFERRED = 3


def pick_movies_to_rate(rng: random.Random, profile: Dict, n: int) -> List[str]:
    """
    Selectionne n films a noter (biaises vers les genres preferes).
    Reserve RESERVED_PREFERRED films preferes pour eviter l'epuisement du pool.
    """
    preferred = []
    other = []
    for mid, genre in MOVIE_CATALOG.items():
        if genre in profile["preferred_genres"]:
            preferred.append(mid)
        else:
            other.append(mid)
    rng.shuffle(preferred)
    rng.shuffle(other)

    # Reserve les derniers RESERVED_PREFERRED films preferes
    available_preferred = preferred[:-RESERVED_PREFERRED] if len(preferred) > RESERVED_PREFERRED else []
    reserved_set = set(preferred[len(available_preferred):])

    selection: List[str] = []
    pi, oi = 0, 0
    for _ in range(n):
        pick_preferred = rng.random() < profile["rating_bias"]
        if pick_preferred and pi < len(available_preferred):
            selection.append(available_preferred[pi])
            pi += 1
        elif oi < len(other):
            selection.append(other[oi])
            oi += 1
        elif pi < len(available_preferred):
            selection.append(available_preferred[pi])
            pi += 1
    return selection


# --- Client API ---
class APIClient:
    """Thin wrapper around the PRISM CINE REST API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self) -> Dict:
        r = self.session.post(f"{self.base}/api/reset", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def set_config(self, cb_weight: Optional[float] = None,
                   epsilon: Optional[float] = None,
                   use_bias: Optional[bool] = None) -> Dict:
        payload: Dict[str, Any] = {}
        if cb_weight is not None:
            payload["cb_weight"] = cb_weight
        if epsilon is not None:
            payload["epsilon"] = epsilon
        if use_bias is not None:
            payload["use_bias"] = use_bias
        r = self.session.post(f"{self.base}/api/config", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def rate(self, user_id: str, movie_id: str, rating: float) -> Dict:
        r = self.session.post(
            f"{self.base}/api/rate",
            json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def recommend(self, user_id: str, n: int = 5) -> Tuple[Dict, float]:
        """Returns (response_json, latency_ms)."""
        t0 = time.perf_counter()
        r = self.session.get(
            f"{self.base}/api/recommend/{user_id}",
            params={"n": n},
            timeout=TIMEOUT,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        return r.json(), latency_ms

    def get_stats(self) -> Dict:
        r = self.session.get(f"{self.base}/api/stats", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()


# --- Calcul des metriques ---
def compute_metrics(
    recs: List[Dict],
    ground_truth: Dict[str, float],
    preferred_genres: List[str],
) -> Dict[str, float]:
    """MAE, GHR (Genre Hit Rate) et couverture de genres."""
    if not recs:
        return {"mae": float("nan"), "ghr": 0.0, "coverage": 0}

    # MAE
    errors = []
    for r in recs:
        mid = r["movie_id"]
        predicted = r.get("predicted_score", 0.0)
        actual = ground_truth.get(mid, 3.0)
        errors.append(abs(predicted - actual))
    mae = float(np.mean(errors)) if errors else float("nan")

    # GHR (sur les MATCH uniquement)
    match_recs = [r for r in recs if r.get("strategy") == "MATCH"]
    if match_recs:
        hits = sum(1 for r in match_recs if MOVIE_CATALOG.get(r["movie_id"]) in preferred_genres)
        ghr = hits / len(match_recs)
    else:
        hits = sum(1 for r in recs if MOVIE_CATALOG.get(r["movie_id"]) in preferred_genres)
        ghr = hits / len(recs)

    genres_seen = set(MOVIE_CATALOG.get(r["movie_id"], "?") for r in recs)
    coverage = len(genres_seen)

    return {"mae": round(mae, 4), "ghr": round(ghr, 4), "coverage": coverage}


# --- Execution du benchmark ---
def run_approach(
    api: APIClient,
    approach: Dict,
    users: Dict[str, Dict],
    max_votes: int,
    checkpoints: List[int],
) -> List[Dict]:
    """Execute une approche : reset, configure, rate, mesure aux checkpoints."""
    name = approach["name"]
    print(f"\n{'='*60}")
    print(f"  APPROACH: {name}")
    print(f"  Config: cb_weight={approach['cb_weight']}, "
          f"epsilon={approach['epsilon']}, use_bias={approach['use_bias']}")
    print(f"{'='*60}")

    api.reset()
    time.sleep(0.3)
    api.set_config(
        cb_weight=approach["cb_weight"],
        epsilon=approach["epsilon"],
        use_bias=approach["use_bias"],
    )

    checkpoint_metrics: Dict[int, List[Dict]] = {cp: [] for cp in checkpoints}
    checkpoint_latencies: Dict[int, List[float]] = {cp: [] for cp in checkpoints}

    for user_id, profile in users.items():
        user_rng = random.Random(f"{SEED}_{user_id}")
        ground_truth = generate_ground_truth(user_rng, profile)
        movies_to_rate = pick_movies_to_rate(user_rng, profile, max_votes)

        for vote_idx, mid in enumerate(movies_to_rate, start=1):
            rating = ground_truth[mid]
            api.rate(user_id, mid, rating)

            if vote_idx in checkpoints:
                recs_resp, latency = api.recommend(user_id, n=5)
                recs = recs_resp.get("recommendations", [])
                m = compute_metrics(recs, ground_truth, profile["preferred_genres"])
                checkpoint_metrics[vote_idx].append(m)
                checkpoint_latencies[vote_idx].append(latency)

    # Agregation par checkpoint
    results = []
    for cp in checkpoints:
        user_metrics = checkpoint_metrics[cp]
        latencies = checkpoint_latencies[cp]

        if not user_metrics:
            continue

        avg_mae = float(np.nanmean([m["mae"] for m in user_metrics]))
        avg_ghr = float(np.mean([m["ghr"] for m in user_metrics]))
        all_coverage = set()
        for m in user_metrics:
            pass
        avg_coverage = float(np.mean([m["coverage"] for m in user_metrics]))
        med_latency = float(np.median(latencies))

        row = {
            "approach": name,
            "votes": cp,
            "mae": round(avg_mae, 4),
            "ghr": round(avg_ghr, 4),
            "coverage": round(avg_coverage, 1),
            "latency_ms": round(med_latency, 2),
        }
        results.append(row)
        print(f"  [{name}] votes={cp:>2}  MAE={row['mae']:.4f}  "
              f"GHR={row['ghr']:.4f}  Cov={row['coverage']:.1f}  "
              f"Lat={row['latency_ms']:.1f}ms")

    return results


def run_alpha_sweep(
    api: APIClient,
    users: Dict[str, Dict],
    max_votes: int,
    alpha_values: List[float],
) -> List[Dict]:
    """Balayage de alpha_cb avec epsilon=0.2, use_bias=True (10 votes/user)."""
    SWEEP_VOTES = 10  # warm-start: enough signal, preferred movies still available

    print(f"\n{'='*60}")
    print(f"  ALPHA SWEEP (GHR vs alpha_cb) — {SWEEP_VOTES} votes per user")
    print(f"  Values: {alpha_values}")
    print(f"{'='*60}")

    results = []
    for alpha in alpha_values:
        api.reset()
        time.sleep(0.3)
        api.set_config(cb_weight=alpha, epsilon=0.2, use_bias=True)

        ghrs: List[float] = []
        maes: List[float] = []
        for user_id, profile in users.items():
            user_rng = random.Random(f"{SEED}_{user_id}")
            ground_truth = generate_ground_truth(user_rng, profile)
            movies_to_rate = pick_movies_to_rate(user_rng, profile, SWEEP_VOTES)

            for mid in movies_to_rate:
                api.rate(user_id, mid, ground_truth[mid])

            recs_resp, _ = api.recommend(user_id, n=5)
            recs = recs_resp.get("recommendations", [])
            m = compute_metrics(recs, ground_truth, profile["preferred_genres"])
            ghrs.append(m["ghr"])
            maes.append(m["mae"])

        avg_ghr = float(np.mean(ghrs))
        avg_mae = float(np.nanmean(maes))
        row = {"alpha_cb": alpha, "ghr": round(avg_ghr, 4), "mae": round(avg_mae, 4)}
        results.append(row)
        print(f"  alpha_cb={alpha:.2f}  MAE={row['mae']:.4f}  GHR={row['ghr']:.4f}")

    return results


# --- Sorties : table, CSV, plots ---
def print_table(rows: List[Dict]) -> None:
    """Pretty-print the comparison table to stdout."""
    header = f"{'Approach':<24} {'Votes':>5} {'MAE':>7} {'GHR':>7} {'Cov':>5} {'Lat(ms)':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['approach']:<24} {r['votes']:>5} {r['mae']:>7.4f} "
              f"{r['ghr']:>7.4f} {r['coverage']:>5.1f} {r['latency_ms']:>8.2f}")
    print(sep)


def write_csv(rows: List[Dict], path: str) -> None:
    """Export results to CSV."""
    fieldnames = ["approach", "votes", "mae", "ghr", "coverage", "latency_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {path}")


def plot_mae_vs_votes(rows: List[Dict], path: str) -> None:
    """Generate MAE vs number-of-votes line plot (one line per approach)."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping {path} (matplotlib not installed)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    approaches_seen: Dict[str, Tuple[List[int], List[float]]] = {}
    for r in rows:
        name = r["approach"]
        if name not in approaches_seen:
            approaches_seen[name] = ([], [])
        approaches_seen[name][0].append(r["votes"])
        approaches_seen[name][1].append(r["mae"])

    markers = ["o", "s", "^", "D", "v"]
    for i, (name, (votes, maes)) in enumerate(approaches_seen.items()):
        marker = markers[i % len(markers)]
        ax.plot(votes, maes, marker=marker, linewidth=2, markersize=8, label=name)

    ax.set_xlabel("Number of Votes per User", fontsize=12)
    ax.set_ylabel("MAE (Mean Absolute Error)", fontsize=12)
    ax.set_title("MAE vs Number of Votes — Approach Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


def plot_ghr_vs_alpha(alpha_rows: List[Dict], path: str) -> None:
    """Graphique double axe : MAE (ligne) + GHR (barres) vs alpha_cb."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping {path} (matplotlib not installed)")
        return

    alphas = [r["alpha_cb"] for r in alpha_rows]
    ghrs = [r["ghr"] for r in alpha_rows]
    maes = [r["mae"] for r in alpha_rows]
    x_labels = [f"{a:.2f}" for a in alphas]
    x_pos = np.arange(len(alphas))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Bars: GHR ---
    bar_colors = ["#4a90d9" if a != 0.7 else "#e74c3c" for a in alphas]
    bars = ax1.bar(x_pos - 0.15, ghrs, width=0.3, color=bar_colors,
                   edgecolor="white", linewidth=1.5, label="GHR", alpha=0.85)
    ax1.set_xlabel("alpha_cb (Content-Based Weight)", fontsize=12)
    ax1.set_ylabel("Genre Hit Rate (GHR)", fontsize=12, color="#4a90d9")
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(axis="y", labelcolor="#4a90d9")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)

    # Annotate
    for bar, ghr in zip(bars, ghrs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{ghr:.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#4a90d9")

    # --- Line: MAE ---
    ax2 = ax1.twinx()
    ax2.plot(x_pos, maes, color="#2ecc71", marker="D", linewidth=2.5,
             markersize=9, label="MAE", zorder=5)
    ax2.set_ylabel("MAE (Mean Absolute Error)", fontsize=12, color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    for i, mae in enumerate(maes):
        ax2.annotate(f"{mae:.3f}", xy=(x_pos[i], mae),
                     xytext=(0, 12), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#27ae60")

    # Marquer la config actuelle
    for i, a in enumerate(alphas):
        if a == 0.7:
            ax1.axvline(x=x_pos[i], color="#c0392b", linestyle="--",
                        linewidth=1.5, alpha=0.6)
            ax1.text(x_pos[i] + 0.05, 1.08, "current\nconfig",
                     fontsize=9, color="#c0392b", fontweight="bold",
                     ha="left")

    # Annotation du minimum MAE
    min_idx = int(np.argmin(maes))
    ax2.plot(x_pos[min_idx], maes[min_idx], "o", markersize=16,
             markerfacecolor="none", markeredgecolor="#27ae60",
             markeredgewidth=2.5, zorder=6)

    ax1.set_title("MAE + GHR vs alpha_cb", fontsize=14)
    ax1.grid(axis="y", alpha=0.2)

    # Legende combinee
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="PRISM CINE — Benchmark comparison of recommendation approaches"
    )
    parser.add_argument(
        "--url", default=BASE_URL_DEFAULT,
        help=f"Base URL of the PRISM CINE server (default: {BASE_URL_DEFAULT})"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast demo mode: 2 users, 10 votes (instead of 5 users, 20 votes)"
    )
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Select users and vote count based on mode
    all_users = list(USER_PROFILES.keys())
    if args.quick:
        n_users = 2
        max_votes = 10
        checkpoints = [5, 10]
        print("MODE: --quick (2 users, 10 votes)")
    else:
        n_users = 5
        max_votes = 20
        checkpoints = [5, 10, 20]
        print("MODE: full benchmark (5 users, 20 votes)")

    selected_users = {uid: USER_PROFILES[uid] for uid in all_users[:n_users]}
    print(f"Users: {list(selected_users.keys())}")
    print(f"Checkpoints: {checkpoints}")

    api = APIClient(args.url)

    # Verify connectivity
    try:
        stats = api.get_stats()
        print(f"Server OK — {stats['total_movies']} movies loaded\n")
    except Exception as e:
        sys.exit(f"ERROR: Cannot reach server at {args.url}\n{e}")

    # Phase 1 : toutes les approches
    all_results: List[Dict] = []
    for approach in APPROACHES:
        results = run_approach(api, approach, selected_users, max_votes, checkpoints)
        all_results.extend(results)

    # Phase 2 : balayage alpha
    alpha_results = run_alpha_sweep(api, selected_users, max_votes, ALPHA_SWEEP_VALUES)

    # Restaurer la config par defaut
    api.reset()
    api.set_config(cb_weight=0.7, epsilon=0.2, use_bias=True)
    print("\nDefault config restored (cb_weight=0.7, epsilon=0.2, use_bias=True)")

    # ---- Output ----
    print_table(all_results)
    write_csv(all_results, "benchmark_results.csv")
    plot_mae_vs_votes(all_results, "mae_vs_votes.png")
    plot_ghr_vs_alpha(alpha_results, "ghr_vs_alpha.png")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
