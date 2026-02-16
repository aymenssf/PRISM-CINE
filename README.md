# PRISM CINE V2

A hybrid recommendation system combining Singular Value Decomposition (SVD) matrix factorization with epsilon-greedy multi-armed bandit strategies for cold-start problem mitigation.

![Web App Screenshot](image.png)

## Overview

PRISM CINE V2 is a movie recommendation engine built as an educational project to demonstrate the integration of collaborative filtering and exploration-exploitation trade-offs in recommendation systems. The system employs a distributed architecture using Ray for stateful actors, Flask for the web interface, and JADE agents for trend detection via XMPP messaging.

### Key Features

- **Hybrid SVD + Content-Based Filtering**: Combines latent factor models with genre affinity scoring
- **Multi-Armed Bandit**: ε-greedy strategy (ε=0.2) balances personalized recommendations with discovery
- **Online Learning**: Incremental SGD updates after each rating, no batch retraining required
- **Distributed Architecture**: Ray actors for concurrent user sessions, JADE agents for asynchronous trend monitoring
- **Cold-Start Handling**: Pure exploration for new users, gradual transition to exploitation as ratings accumulate

### Technical Implementation

**Core Algorithm:**
```
prediction = μ + b_u + b_i + P_u · Q_i^T
```
where:
- μ = global mean rating
- b_u, b_i = user and item biases
- P_u, Q_i = k-dimensional latent factor vectors (k=10)

**Hybrid Scoring:**
```
final_score = (1 - α) · SVD_score + α · CB_score
```
with α=0.7 weighting toward content-based genre affinity.

**Update Strategy:**
- 10 SGD epochs per rating with learning rate η=0.05
- First epoch updates both user and item factors
- Subsequent epochs update user factors only (prevents cross-contamination)
- L2 regularization λ=0.02

## Architecture

```
┌───────────────┐
│   Browser     │
│  (User UI)    │
└───────┬───────┘
        │ HTTP
        ↓
┌───────────────────────────────────────┐
│  Flask Web Server (Port 5000)        │
│  ├─ Routes (/api/rate, /api/reco)   │
│  └─ TMDB Client (poster fetching)    │
└───────┬───────────────┬───────────────┘
        │ Ray RPC       │ XMPP (C2S)
        ↓               ↓
┌───────────────┐   ┌──────────────┐
│  Ray Actor    │   │  ejabberd    │
│  Recommender  │   │  XMPP Broker │
│  System       │   └──────┬───────┘
└───────────────┘          │ XMPP (S2S)
                           ↓
                   ┌───────────────┐
                   │  JADE Agent   │
                   │  TrendScout   │
                   │  (Genre Boost)│
                   └───────────────┘
```

### Components

1. **Flask Application** (`app/`): Web server exposing REST API and UI
2. **Ray RecommenderSystem Actor** (`app/core/recommender.py`): Stateful SVD model with online updates
3. **JADE TrendScout Agent** (`java_agent/`): Monitors genre popularity, broadcasts boost signals
4. **ejabberd**: XMPP message broker for agent communication
5. **TMDB Client** (`app/utils/tmdb_client.py`): Fetches movie posters via The Movie Database API

## Quick Start

### Prerequisites

- Docker & Docker Compose
- TMDB API key (free): https://www.themoviedb.org/settings/api

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PRISM CINE
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

3. Start services:
```bash
docker-compose up --build
```

4. Populate demo data (optional):
```bash
python demo_fill.py --users 5 --ratings 80
```

5. Open browser:
```
http://localhost:5000
```

## Project Structure

```
PRISM CINE/
├── app/
│   ├── core/
│   │   └── recommender.py       # SVD + Bandit implementation
│   ├── templates/
│   │   └── index.html            # Web UI
│   ├── utils/
│   │   └── tmdb_client.py        # TMDB API integration
│   ├── app.py                    # Flask routes
│   └── config.py                 # Hyperparameters
├── java_agent/                   # JADE TrendScout agent
├── ejabberd/                     # XMPP configuration
├── demo_fill.py                  # Demo data generator
├── test_exhaustif.py             # Test suite
├── docker-compose.yml            # Service orchestration
└── README_DEMO.md                # Detailed demo guide
```

## Configuration

Key hyperparameters in `app/config.py` and `app/core/recommender.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LATENT_FACTORS_K` | 10 | Dimensionality of latent space |
| `LEARNING_RATE` | 0.05 | SGD step size |
| `REGULARIZATION` | 0.02 | L2 penalty coefficient |
| `EPSILON` | 0.2 | Exploration probability |
| `SGD_EPOCHS` | 10 | Update iterations per rating |
| `CB_WEIGHT` | 0.7 | Content-based blend factor |
| `GLOBAL_MEAN` | 3.0 | Rating scale center (1-5) |

## Testing

Run comprehensive test suite:
```bash
python test_exhaustif.py
```

Tests cover:
- Cold-start behavior (100% exploration)
- Warm-start recommendations (80% exploitation)
- Genre alignment with user preferences
- SVD convergence after multiple ratings
- Multi-user isolation

## Dataset

The system ships with 50 films across 10 genres (5 per genre):
- Action, Adventure, Animation, Comedy, Drama
- Fantasy, Horror, Romance, Sci-Fi, Thriller

Each film includes:
- TMDB ID for poster fetching
- Genre classification
- Release year

See `app/core/recommender.py` lines 49-119 for full catalog.

## API Endpoints

### POST `/api/rate`
Submit a user rating.

**Request:**
```json
{
  "user_id": "alice",
  "movie_id": "movie_1",
  "rating": 4.5
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Rating recorded for alice"
}
```

### GET `/api/recommend/<user_id>`
Get personalized recommendations.

**Response:**
```json
{
  "recommendations": [
    {
      "movie_id": "movie_42",
      "title": "Blade Runner 2049",
      "predicted_score": 4.2,
      "badge": "MATCH",
      "genres": ["Sci-Fi"]
    }
  ]
}
```

### GET `/api/stats`
System statistics.

**Response:**
```json
{
  "total_users": 5,
  "total_ratings": 80,
  "total_movies": 50,
  "genre_counts": {"Sci-Fi": 15, "Thriller": 13, ...}
}
```

### POST `/api/reset`
Reset all ratings and user data (development only).

## Performance Characteristics

- **Time Complexity**: O(nk) per recommendation where n=movies, k=latent factors
- **Space Complexity**: O(uk + nk) for u users, n items
- **Update Latency**: <50ms per rating (10 SGD epochs)
- **Cold-Start Response**: Immediate (random exploration)

**Scalability vs. Baselines:**
- ✓ Faster than cosine similarity (O(n²))
- ✓ No batch retraining overhead
- ✓ Stateless HTTP API (horizontal scaling ready)

## Known Limitations

1. **Small Dataset**: 50 films for demonstration purposes; real systems require thousands
2. **In-Memory State**: Ray actor state is not persisted; restart loses all ratings
3. **Single-Threaded SGD**: Updates are sequential; parallel updates would require locking
4. **Genre-Only Content**: No metadata like director, actors, year for richer CB features

## Future Enhancements

- Persistent storage (PostgreSQL/Redis for ratings)
- Implicit feedback (views, hover time)
- Temporal dynamics (rating recency weighting)
- Deep learning embeddings (replace hand-crafted genre features)
- A/B testing framework for bandit evaluation

## License

This project is an academic demonstration and is provided as-is for educational purposes.

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30-37.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook* (2nd ed.). Springer.

## Contributing

This is an academic project developed for demonstration purposes. For questions or suggestions, please open an issue.

## Acknowledgments

- The Movie Database (TMDB) for poster images
- ProcessOne for ejabberd XMPP server
- Anyscale for Ray distributed computing framework
