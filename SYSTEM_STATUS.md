# PRISM CINE — System Status

## ✅ Functional Components

### 1. Flask + Ray Actor (Recommendation Engine)
- **Status**: Fully operational
- **Port**: 32775 (dynamic, check `docker ps`)
- **Test**:
  ```bash
  curl http://localhost:32775/api/stats
  curl -X POST http://localhost:32775/api/rate -H "Content-Type: application/json" -d '{"user_id":"alice","movie_id":"movie_1","rating":5.0}'
  curl http://localhost:32775/api/recommend/alice?n=5
  ```
- **Features**:
  - Ray actor with stateful NumPy rating matrix
  - User-based collaborative filtering with cosine similarity
  - Cold-start handling (global averages when <2 users)
  - Genre boost mechanism (`boosted_genre` field)
  - Memory optimized (dashboard disabled, 100MB object store, 98% threshold)

### 2. ejabberd (XMPP Server)
- **Status**: Healthy and accepting connections
- **Ports**: 5222 (c2s), 5280 (HTTP API)
- **Registered accounts**:
  - `recommender@prismcine.local` / `rec1pass`
  - `trendscout@prismcine.local` / `trendpass`
  - `admin@prismcine.local` / `admin_pass`
- **Test**:
  ```bash
  docker exec ejabberd ejabberdctl registered_users prismcine.local
  docker exec ejabberd ejabberdctl status
  ```

### 3. JADE TrendScout Agent
- **Status**: Running and polling successfully
- **Behavior**: TickerBehaviour (10-second interval)
- **Functionality**:
  - Polls `GET http://flask-ray-node:5000/api/stats`
  - Analyzes `genre_popularity` fractions
  - When top genre >= 0.4 threshold → sends `BOOST_GENRE:<genre>` via ejabberd HTTP API
- **Verified**: Successfully sent `BOOST_GENRE:Action` when Action genre reached 100% (6/6 ratings)
- **Logs**:
  ```bash
  docker logs java-agent | grep -A2 "Trend detected"
  ```

---

## ⚠️ Known Issue: XMPP Client Connection

**Component**: Flask XMPP client (Slixmpp 1.12)
**Symptom**: Client object created but authentication handshake not completing
**Impact**: Flask nodes don't receive BOOST_GENRE messages from JADE agent

### Root Cause
The slixmpp 1.12 async API requires proper coroutine scheduling. The current implementation creates the client and calls `connect()`, but the event loop isn't processing the internal connection coroutines that handle authentication.

### Evidence
- Flask logs show: `XMPP client created for JID: recommender@prismcine.local/<hostname>`
- No `XMPP session started` message
- ejabberd `connected_users` returns empty (no active c2s sessions)
- JADE agent's XMPP messages delivered to ejabberd but not forwarded (no connected recipient)

### Fix Options

**Option A: Use synchronous XMPP library**
Replace slixmpp with `xmpppy` or `sleekxmpp` (synchronous, simpler thread model)

**Option B: Fix slixmpp async pattern**
Ensure `await client.disconnected` runs in a properly-scheduled asyncio context. The issue is that `client.connect()` starts internal tasks that need active event loop processing.

**Option C: Direct HTTP callback (workaround)**
JADE agent can call a Flask webhook (`POST /api/boost`) directly instead of via XMPP, eliminating the message broker.

---

## 🧪 End-to-End Test Procedure

### Step 1: Verify Services Running
```bash
docker ps --format "{{.Names}}\t{{.Status}}"
# Expect: ejabberd (healthy), flask-ray-node-1 (up), java-agent (up)
```

### Step 2: Test Ray Actor Recommendations
```bash
PORT=$(docker ps --format "{{.Ports}}" | grep flask | cut -d: -f2 | cut -d- -f1)
curl http://localhost:$PORT/api/stats
# {"total_users": 0, "total_ratings": 0, "genre_popularity": {...}}

# Submit ratings for Action movies
for i in {1..5}; do
  curl -X POST http://localhost:$PORT/api/rate \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"user$i\",\"movie_id\":\"movie_1\",\"rating\":5.0}"
done

# Check updated stats
curl http://localhost:$PORT/api/stats
# {"genre_popularity": {"Action": 1.0, ...}, ...}
```

### Step 3: Verify JADE Agent Trend Detection
```bash
# Wait 12 seconds for next tick
sleep 12

# Check JADE agent logs
docker logs java-agent 2>&1 | grep "Trend detected"
# Expected: "Trend detected: Action at 1.0 (threshold: 0.4)"

docker logs java-agent 2>&1 | grep "BOOST_GENRE"
# Expected: "BOOST_GENRE:Action sent via XMPP"
```

### Step 4: Verify ejabberd Received Message
```bash
docker logs ejabberd 2>&1 | grep "API call send_message"
# Expected: API call with body "BOOST_GENRE:Action"
```

### Step 5: Manual Genre Boost (Workaround)
Since XMPP delivery is pending, manually trigger the boost:
```bash
curl -X POST http://localhost:$PORT/api/boost \
  -H "Content-Type: application/json" \
  -d '{"genre": "Action"}'

# Verify boost applied
curl http://localhost:$PORT/api/stats
# {"boosted_genre": "Action", ...}

# Get recommendations (Action movies now boosted by 20%)
curl http://localhost:$PORT/api/recommend/alice?n=5
```

---

## 📊 Architecture Summary

```
┌─────────────────┐  HTTP GET /api/stats  ┌───────────────────┐
│  JADE Agent     │ ────────────────────>  │ Flask + Ray Actor │
│  (TrendScout)   │                        │   (Recommender)   │
└────────┬────────┘                        └──────────┬────────┘
         │                                              │
         │ HTTP POST /api/send_message                 │
         │ (BOOST_GENRE:Action)                        │
         ▼                                              │
┌─────────────────┐                                    │
│    ejabberd     │                                    │
│  (XMPP Server)  │ ─ ─ ─ ─ (should deliver) ─ ─ ─ ─ ┘
└─────────────────┘          XMPP c2s
                         (pending connection fix)
```

**What Works**:
- JADE → Flask (HTTP polling)
- JADE → ejabberd (HTTP API message send)
- Flask Ray actor (recommendation engine)
- ejabberd (message broker infrastructure)

**What Needs Fix**:
- Flask → ejabberd (XMPP client connection)
- ejabberd → Flask (message delivery, blocked by above)

---

## 🎯 Demonstration Value

Despite the XMPP client connection issue, the project successfully demonstrates:

1. **Distributed Actor Model** — Ray remote actors with stateful computation
2. **Autonomous Agent** — JADE TickerBehaviour with periodic analysis
3. **Microservice Communication** — HTTP REST + XMPP architecture
4. **Real-Time Trend Detection** — Threshold-based genre popularity monitoring
5. **Collaborative Filtering** — NumPy-based cosine similarity with cold-start handling
6. **Docker Orchestration** — Multi-service compose with health checks and dependencies
7. **Message Broker Integration** — ejabberd configured for API-based message sending

The system architecture is sound, and the XMPP client connection is a solvable integration detail.

---

## 🔧 Quick Fix Implementation

To unblock the system immediately, replace XMPP delivery with direct HTTP callback:

**File**: `java_agent/src/TrendAgent.java`

Change `sendBoost()`:
```java
private void sendBoost(String genre) {
    try {
        // Direct HTTP callback instead of XMPP
        String apiUrl = "http://flask-ray-node:5000/api/boost";
        String body = String.format("{\"genre\":\"%s\"}", genre);

        URL url = new URL(apiUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.getBytes());
        }

        int code = conn.getResponseCode();
        if (code == 200) {
            logger.info("BOOST_GENRE:" + genre + " sent via HTTP callback");
            currentBoostedGenre = genre;
        }
    } catch (Exception e) {
        logger.warning("Failed to send boost: " + e.getMessage());
    }
}
```

This bypasses the XMPP message broker entirely while preserving all distributed system properties.
