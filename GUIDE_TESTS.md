# Guide de Test — Kinetoscope V2 (SVD + Bandit)

## Lancement de l'application

```bash
# 1. Lancer les services Docker
docker-compose up --build

# Attendre que tous les services soient démarrés
# Vous devriez voir : [RAY-ACTOR] RecommenderSystem initialised (SVD k=10...)
```

## Tests Automatiques Complets

Le script `test_svd_bandit.py` teste l'ensemble du système :

```bash
# 2. Dans un autre terminal, lancer les tests
pip install requests colorama  # Installation optionnelle de colorama pour les couleurs
python test_svd_bandit.py
```

### Tests couverts

1. **Cold-Start** — Vérification que les recommandations sans données retournent `strategy: "DISCOVERY"`
2. **Premier Rating SGD** — Validation de la mise à jour des facteurs latents
3. **MATCH/DISCOVERY Mix** — Vérification du ratio 80/20 (exploitation/exploration)
4. **Multi-Users Personnalisation** — Alice (Sci-Fi) vs Bob (Horror) ont des recs différentes
5. **Genre Boost** — Simulation JADE : boost Sci-Fi prioritise les films Sci-Fi
6. **Edge Cases** — User inconnu, tous films notés, clé `strategy` présente
7. **SGD Convergence** — Vérification que l'erreur diminue avec les re-ratings

### Sortie attendue

```
╔═══════════════════════════════════════════════════════════════════╗
║         KINETOSCOPE V2 — TEST SUITE (SVD + BANDIT)              ║
╚═══════════════════════════════════════════════════════════════════╝

✓ Serveur Flask accessible a http://localhost:5000

======================================================================
                    TEST 1 : COLD-START (Zero Donnees)
======================================================================

ℹ Users: 0, Ratings: 0
ℹ Cold-start retourne 5 films
✓ Toutes les recs sont marquees DISCOVERY (correct)
→   The Matrix (Action) — Strategy: DISCOVERY, Score: 0.0
→   Mad Max: Fury Road (Action) — Strategy: DISCOVERY, Score: 0.0
→   Interstellar (Sci-Fi) — Strategy: DISCOVERY, Score: 0.0
✓ TEST 1 PASSE

[... 6 autres tests ...]

TOUS LES TESTS REUSSIS (7/7) ! 🎉
Le moteur SVD+Bandit fonctionne correctement.
```

## Test Manuel via UI

1. Ouvrir **http://localhost:5000**
2. Soumettre une note (ex: user_id=`alice`, film=`Interstellar`, score=`5`)
3. Observer les logs Docker → chercher `prediction_error=X.XXXX`
4. Taper `alice` dans "Target User ID" et cliquer **Refresh**
5. Vérifier la présence de :
   - **4 cartes MATCH** : badge vert `Match: 4.2/5` au hover
   - **1 carte DISCOVERY** : badge violet pulsant `🎲 AI DISCOVERY`

## Simulation JADE (Genre Boost)

```bash
# 3. Générer du trafic Sci-Fi pour déclencher JADE
python simulate_traffic.py
```

- Attend 20 votes Sci-Fi
- JADE détecte la tendance
- Bannière verte "Genre en tendance : Sci-Fi" apparaît
- Films Sci-Fi montés dans le classement

## Vérification des Logs

Les logs Docker doivent montrer :

```
[RAY-ACTOR] SVD updated: user=alice -> Interstellar (Sci-Fi) = 5.0  |  prediction_error=3.2415  |  Total votes: 1
[RAY-ACTOR] Recommendations for alice: 5 items (SVD+Bandit: 4 MATCH, 1 DISCOVERY)
```

## Troubleshooting

- **Port 5000 occupé** : Modifier `docker-compose.yml` ligne 67 → `"5001:5000"`
- **Tests échouent** : Vérifier que `docker-compose up` tourne sans erreur
- **Pas de DISCOVERY** : Vérifier `epsilon=0.2` dans `recommender.py` ligne 42
- **Recommandations identiques** : SGD a besoin de plusieurs ratings pour diverger

## Debug Avancé

```bash
# Voir les logs en temps réel
docker-compose logs -f flask-ray-node

# Inspecter la matrice Ray
docker exec -it <container_id> python3
>>> import ray
>>> # Accès aux vecteurs latents P et Q
```

## Performance

- **Cold-start** : <50ms
- **Rating + SGD update** : ~5ms
- **Recommandations (5 items)** : ~10ms
- **Convergence SGD** : ~10-20 ratings par utilisateur

---

**Prêt pour la soutenance !** 🎬
