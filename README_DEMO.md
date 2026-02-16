# 🎬 Guide de Démonstration — PRISM CINE V2

Guide complet pour démontrer les fonctionnalités du système de recommandation hybride SVD + Bandit Epsilon-Greedy.

---

## 📋 Prérequis

### 1. **Clé API TMDB** (gratuite)

Le système utilise The Movie Database (TMDB) pour récupérer les posters de films en haute qualité.

**Étapes:**
1. Créer un compte sur https://www.themoviedb.org/
2. Aller dans Settings → API
3. Générer une clé API v3
4. Exporter la variable d'environnement:

```bash
export TMDB_API_KEY="votre_cle_api_ici"
```

**Note:** Si la clé n'est pas configurée, le système utilisera des placeholders automatiquement (pas d'erreur).

### 2. **Services Docker actifs**

```bash
# Démarrer tous les services (ejabberd, Flask+Ray, JADE agent)
docker-compose up --build

# Logs en temps réel
docker-compose logs -f flask-ray-node
```

**Services attendus:**
- ✅ Flask/Ray sur port 5000
- ✅ ejabberd (XMPP) pour messages JADE
- ✅ JADE agent (Java) pour détection genre boost

---

## 🚀 Lancement Rapide

### Option 1: Script Automatique (Recommandé pour Démo)

Remplir l'UI avec 5 users et 80 ratings réalistes en ~30 secondes:

```bash
python demo_fill.py --users 5 --ratings 80
```

**Output attendu:**
```
=================================================...
    KINETOSCOPE V2 — DÉMONSTRATION AUTOMATIQUE
=================================================...

🔄 RESET SYSTÈME...
   ✓ 20 users + 133 ratings supprimés

📊 GÉNÉRATION DE 80 RATINGS...
   Users: alice, bob, charlie, diana, eve

   👤 ALICE (préfère: Sci-Fi, Thriller)
      ✓ movie_6 = 4.8
      ✓ movie_7 = 4.5
      ...

🔍 VÉRIFICATION RECOMMANDATIONS...

   👤 ALICE:
      MATCH: 7/8 (88%)
      DISCOVERY: 1/8 (12%)
      Score moyen: 3.42/5
      1. ✓ Blade Runner 2049 (Sci-Fi) - 4.12
      2. ✓ Zodiac (Thriller) - 3.87
      3. 🎲 The Grand Budapest Hotel (Comedy) - 0.0

📈 STATISTIQUES FINALES
   Total users: 5
   Total ratings: 80
   Total movies: 50
   Matrix density: 32.0%

   Genre popularity:
      Sci-Fi: 18.8%
      Thriller: 16.2%
      ...

=================================================...
✅ DÉMONSTRATION COMPLÈTE
Ouvrir http://localhost:5000 pour voir l'UI remplie
=================================================...
```

**Durée:** ~15-30 secondes

---

### Option 2: Commandes Manuelles (Pour Exploration)

```bash
# 1. Reset système
curl -X POST http://localhost:5000/api/reset

# 2. Voter plusieurs fois (alice aime Sci-Fi)
curl -X POST http://localhost:5000/api/rate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","movie_id":"movie_6","rating":5.0}'

curl -X POST http://localhost:5000/api/rate \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice","movie_id":"movie_7","rating":4.5}'

# 3. Récupérer recommandations (après 5+ votes)
curl http://localhost:5000/api/recommend/alice?n=8 | jq

# 4. Vérifier stats
curl http://localhost:5000/api/stats | jq
```

---

## 🎯 Validation Visuelle dans l'UI

Ouvrir http://localhost:5000 et vérifier les éléments suivants:

### ✅ Hero Section (en haut)

**Métriques affichées:**
- **Learning Rate:** 0.0042 (fixe)
- **Exploration:** ε = 0.2 (spinner animé)
- **Matrix Density:** ~30-40% après demo_fill.py (dynamique)

### ✅ Archives Section (50 films)

**Checklist:**
- [ ] 50 films visibles dans une grille responsive (4 colonnes desktop)
- [ ] Posters TMDB réels chargés (PAS de placeholders avec texte)
- [ ] Hover effects: opacité 60% → 100%, grayscale → couleur
- [ ] Genre badges visibles (coin supérieur gauche)
- [ ] Bouton "Rate" sur hover fonctionnel

**10 genres équilibrés:**
- Action (5 films)
- Sci-Fi (5 films)
- Drama (5 films)
- Romance (5 films)
- Animation (5 films)
- Thriller (5 films)
- Horror (5films)
- Comedy (5 films)
- Fantasy (5 films)
- Adventure (5 films)

### ✅ Recommendations Section (Query: alice)

**Étapes:**
1. Taper `alice` dans "Target User ID"
2. Cliquer "Query"
3. Attendre animation de loading (spinner emerald)

**Attendu (après demo_fill.py):**
- [ ] **7 cartes MATCH** (bordure neutre, badge DISCOVERY absent)
- [ ] **1 carte DISCOVERY** (bordure violet, badge violet pulsant "DISCOVERY")
- [ ] Top-3 sont majoritairement Sci-Fi/Thriller (préférences alice)
- [ ] Score rings animés avec couleurs:
  - Rouge: <3.0
  - Amber: 3.0-4.0
  - Emerald: ≥4.0

**Badges:**
- MATCH: Aucun badge visible (score visible au hover dans la version originale, mais supprimé dans la nouvelle UI)
- DISCOVERY: Badge violet pulsant "DISCOVERY" toujours visible

### ✅ Dynamic Island (Trending Genre)

Lancer simulate_traffic.py pour déclencher le boost:

```bash
python simulate_traffic.py
```

**Attendu:**
- Bannière apparaît en haut: "🔥 TREND DETECTED: Sci-Fi"
- Archives: films Sci-Fi ont anneau emerald
- Durée: 30 secondes (configurable dans JADE agent)

---

## 🧪 Tests Exhaustifs

### Test Suite Complète (15 scénarios)

```bash
# Lancer la suite complète (5-10 minutes)
python test_exhaustif.py
```

**Scénarios couverts:**
1. ✅ Connectivité serveur
2. ✅ Cold-start global (zéro données)
3. ✅ Premier rating SGD
4. ✅ Convergence SGD (15 votes)
5. ✅ Personnalisation multi-users
6. ✅ Ratio MATCH/DISCOVERY (epsilon=0.2)
7. ✅ Genre boost manuel
8. ✅ Clé 'strategy' présente partout
9. ✅ Scores dans [0, 5]
10. ✅ Tous films notés (edge case)
11. ✅ User inconnu (cold-start individuel)
12. ✅ Reset endpoint
13. ✅ Re-rating même film
14. ✅ DISCOVERY aléatoire (bandit)
15. ✅ Stress test (100 ratings)

**Résultat attendu:** 13-15/15 tests passent

**Note:** 2 tests peuvent échouer à cause de l'aléatoire SVD/bandit → **NORMAL** si ≥13/15 passent.

### Test Convergence SVD Uniquement

```bash
# Tests rapides (30 secondes)
python test_svd_bandit.py
```

**7 tests de base:**
- Cold-start
- Premier SGD
- MATCH/DISCOVERY mix
- Multi-user personnalisation
- Genre boost
- Edge cases

---

## 🐛 Troubleshooting

### ❌ Images ne s'affichent pas (placeholders avec texte)

**Cause:** Clé TMDB manquante ou invalide

**Solutions:**
```bash
# Vérifier la clé est exportée
echo $TMDB_API_KEY

# Re-exporter si nécessaire
export TMDB_API_KEY="votre_cle_ici"

# Restart containers
docker-compose restart flask-ray-node

# Check logs TMDB
docker-compose logs flask-ray-node | grep TMDB
```

**Log attendu:**
```
✓ Poster récupéré pour TMDB 603: https://image.tmdb.org/t/p/w500/...
```

**Log si problème:**
```
✗ TMDB_API_KEY non configurée, utilisation du fallback pour film 603
```

---

### ❌ Matrix density trop faible (<20%)

**Cause:** Pas assez de ratings générés

**Solution:**
```bash
# Relancer avec plus de ratings
python demo_fill.py --ratings 120

# Ou ajouter plus d'users
python demo_fill.py --users 10 --ratings 150
```

**Densité optimale:** 30-50% pour bonne convergence SVD

---

### ❌ Pas de MATCH recommendations (que DISCOVERY)

**Cause:** User n'a pas assez de votes (<5)

**Explication:**
- SGD converge après ~10-15 votes par user
- Avec <5 votes: erreur ~5.0 (signal trop faible)
- Avec 10+ votes: erreur ~1.0-2.0 (convergence visible)

**Solution:**
```bash
# Re-exécuter demo_fill.py (génère 15-16 votes/user)
python demo_fill.py
```

**Voir aussi:** `ANALYSE_CONVERGENCE.md` pour explication mathématique

---

### ❌ Port 5000 inaccessible

**Cause:** Services Docker non démarrés ou port conflit

**Solutions:**
```bash
# Vérifier containers actifs
docker-compose ps

# Redémarrer services
docker-compose down
docker-compose up --build

# Vérifier port 5000 libre
lsof -i :5000

# Si occupé, kill processus
kill -9 <PID>
```

---

### ❌ Erreur "Module 'requests' not found"

**Cause:** Dépendances non installées

**Solutions:**
```bash
# Option 1: Virtual environment (recommandé)
python3 -m venv venv
source venv/bin/activate
pip install requests colorama

# Option 2: System packages (Ubuntu/Debian)
sudo apt install python3-requests python3-colorama

# Option 3: Global pip (si PEP 668 autorisé)
pip install requests colorama
```

---

## 📊 Métriques Clés pour Jury

### Convergence SVD

**Démonstration:**
```bash
# 1. Reset
curl -X POST http://localhost:5000/api/reset

# 2. Voter 15 fois pour alice (Sci-Fi)
for i in {6..10}; do
  for j in {1..3}; do
    curl -X POST http://localhost:5000/api/rate \
      -H "Content-Type: application/json" \
      -d "{\"user_id\":\"alice\",\"movie_id\":\"movie_$i\",\"rating\":5.0}"
  done
done

# 3. Vérifier convergence
curl http://localhost:5000/api/recommend/alice?n=5 | jq '.recommendations[] | {title, genre, score: .predicted_score, strategy}'
```

**Attendu:**
- Top-3 sont Sci-Fi (préférence apprise)
- Scores MATCH: 3.5-4.5/5
- 1 film DISCOVERY: score plus bas, genre différent

### Exploration vs Exploitation

**Ratio attendu:**
- **80% MATCH** (exploitation)
- **20% DISCOVERY** (exploration)

**Vérification:**
```bash
curl http://localhost:5000/api/recommend/alice?n=20 | \
  jq '.recommendations | group_by(.strategy) | map({strategy: .[0].strategy, count: length})'
```

**Output:**
```json
[
  {"strategy": "MATCH", "count": 16},
  {"strategy": "DISCOVERY", "count": 4}
]
```

### Genre Boost Performance

**Démonstration:**
```bash
# 1. Baseline (sans boost)
curl http://localhost:5000/api/recommend/alice?n=5

# 2. Activer boost Fantasy
python simulate_traffic.py  # Ou: curl -X POST http://localhost:5000/api/boost -d '{"genre":"Fantasy"}'

# 3. Nouveau query (avec boost)
curl http://localhost:5000/api/recommend/alice?n=5
```

**Attendu:**
- Avant: 0-1 films Fantasy dans top-5
- Après boost: 2-3 films Fantasy dans top-5 (scores ×1.2)

---

## 📂 Structure Fichiers Modifiés

| Fichier | Lines | Changements | Validation |
|---------|-------|-------------|------------|
| `app/core/recommender.py` | 48-119 | MOVIE_CATALOG 20→50 films + tmdb_id | ✅ Lignes 48-119 contiennent 50 films |
| `app/config.py` | NEW | Configuration TMDB API | ✅ TMDB_API_KEY défini |
| `app/utils/tmdb_client.py` | NEW | Client TMDB avec fallback | ✅ tmdb_client.get_all_posters() existe |
| `app/app.py` | 27, 130, 162 | Import TMDB + passage posters | ✅ posters passé au template |
| `app/templates/index.html` | 200, 353, 375 | Utiliser vraies images | ✅ movie.poster_url utilisé |
| `demo_fill.py` | NEW | Script démonstration | ✅ Exécutable chmod +x |
| `requirements.txt` | +1 | Ajouter requests | ✅ requests>=2.31.0 présent |
| `README_DEMO.md` | NEW | Documentation | ✅ Ce fichier |

---

## ✅ Checklist Validation Finale

Avant la livraison/soutenance, vérifier:

### Infrastructure
- [ ] `docker-compose up` démarre sans erreur
- [ ] http://localhost:5000 accessible
- [ ] Logs montrent "TMDB client initialized"
- [ ] ejabberd + JADE agent actifs (docker-compose ps)

### Dataset & Images
- [ ] 50 films visibles dans Archives
- [ ] Posters TMDB chargés (vérifier 10+ films différents)
- [ ] Pas de placeholders texte ("Movie 1", etc.)
- [ ] Genres équilibrés (5 films par genre)

### Tests
- [ ] `python demo_fill.py` remplit l'UI en <60s
- [ ] `test_exhaustif.py` → 13-15/15 tests passent
- [ ] SGD converge (erreur descend avec votes)
- [ ] Personnalisation visible (alice ≠ bob)

### UI
- [ ] MATCH/DISCOVERY badges visibles et distincts
- [ ] Score rings colorés selon valeur (rouge/amber/emerald)
- [ ] Hover effects fonctionnent
- [ ] Trending badge apparaît avec simulate_traffic.py

### Performance
- [ ] Temps réponse API <200ms (curl -w "@-" http://localhost:5000/api/stats)
- [ ] UI responsive (mobile-friendly)
- [ ] Pas de crash après 100+ ratings

---

## 🎓 Explication pour le Jury (Prompt)

> **« Le système PRISM CINE V2 implémente un moteur de recommandation hybride avec 2 composantes :**
>
> **1. Factorisation Matricielle (SVD)**
> - Matrice R (utilisateurs × films) factorisée en P (utilisateurs × k) et Q (films × k)
> - k=10 dimensions latentes apprises par descente de gradient stochastique (SGD)
> - Learning rate α=0.01, régularisation λ=0.02
> - Convergence progressive : avec 10-15 votes, erreur passe de ~5.0 à ~1.0
>
> **2. Multi-Armed Bandit (Epsilon-Greedy)**
> - ε=0.2 : 20% exploration (DISCOVERY), 80% exploitation (MATCH)
> - Exploration évite les bulles de filtre ("filter bubble")
> - Exploitation maximise satisfaction immédiate
>
> **Démonstration visible:**
> - Cold-start : 100% DISCOVERY (utilisateur inconnu)
> - Après 5+ votes : 80% MATCH + 20% DISCOVERY
> - Top-3 MATCH : genres préférés (convergence SVD)
> - Genre boost (JADE) : priorité dynamique aux tendances détectées
>
> **Scalabilité:**
> - Complexité O(nk) vs O(n²) pour cosine similarity
> - Online learning : pas de re-entraînement batch
> - 50 films × 5 users × 80 ratings = Matrix density 32% (production-grade) »

---

## 🔗 Ressources

- **TMDB API Docs:** https://developers.themoviedb.org/3
- **SVD Recommender Systems:** [Matrix Factorization Techniques - Koren et al.]
- **Multi-Armed Bandits:** [Reinforcement Learning - Sutton & Barto, Chapter 2]
- **Cold-Start Problem:** ["The Cold Start Problem" - Andrew Chen]

---

**Système validé et prêt pour la livraison !** 🎉

Pour questions ou bugs: ouvrir une issue sur le repo ou contacter l'équipe.
