import jade.core.Agent;
import jade.core.behaviours.TickerBehaviour;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

/**
 * TrendScout -- Agent JADE de detection autonome de tendances par genre.
 *
 * Ici on utilise JADE car :
 * JADE fournit un cycle de vie formel d agent (setup / behaviours / takeDown)
 * qui separe proprement la logique de detection (TrendScoutBehaviour) du
 * mecanisme de planification (TickerBehaviour a intervalles de 10s).
 * Cela decouple entierement l analyse de tendances du cycle requete Flask.
 *
 * Pourquoi le polling periodique (pas evenementiel) :
 * La detection de tendances est une operation batch sur les distributions
 * agregees de notes. Un polling toutes les 10s equilibre la reactivite
 * avec la charge serveur pour un systeme de recommandation ou les
 * tendances evoluent lentement.
 *
 * Chemin de communication :
 * TrendScout --HTTP GET--> Flask /api/stats
 * TrendScout --HTTP POST-> ejabberd /api/send_message (BOOST_GENRE)
 * ejabberd --XMPP c2s--> Flask Slixmpp clients (tous les replicas)
 *
 * Seuil : Un genre avec > 5 votes absolus declenche le boost (seuil bas
 * pour la demo en soutenance).
 */
public class TrendAgent extends Agent {

    // -- ANSI Color codes pour des logs terminaux colores et visibles --
    private static final String RESET = "\033[0m";
    private static final String BOLD = "\033[1m";
    private static final String CYAN = "\033[36m";
    private static final String GREEN = "\033[32m";
    private static final String YELLOW = "\033[33m";
    private static final String RED = "\033[31m";
    private static final String MAGENTA = "\033[35m";
    private static final String BLUE = "\033[34m";
    private static final String WHITE = "\033[97m";

    private static final String TAG = BOLD + MAGENTA + "[JADE-INTEL]" + RESET + " ";

    private String flaskServiceUrl;
    private String xmppHost;
    private String xmppAdminUser;
    private String xmppAdminPass;
    private int voteThreshold; // Seuil en nombre absolu de votes
    private String currentBoostedGenre = null;
    private int scanCount = 0;

    // ------------------------------------------------------------------
    // Cycle de vie de l agent
    // ------------------------------------------------------------------

    @Override
    protected void setup() {
        flaskServiceUrl = envOrDefault("FLASK_SERVICE_URL", "http://flask-ray-node:5000");
        xmppHost = envOrDefault("XMPP_HOST", "ejabberd");
        xmppAdminUser = envOrDefault("XMPP_ADMIN_USER", "admin");
        xmppAdminPass = envOrDefault("XMPP_ADMIN_PASS", "admin_pass");
        voteThreshold = Integer.parseInt(envOrDefault("VOTE_THRESHOLD", "5"));

        System.out.println(TAG + GREEN + "=============================================" + RESET);
        System.out.println(TAG + GREEN + "  TrendScout Agent STARTING" + RESET);
        System.out.println(TAG + CYAN + "  Target  : " + WHITE + flaskServiceUrl + RESET);
        System.out.println(TAG + CYAN + "  Seuil   : " + WHITE + "> " + voteThreshold + " votes" + RESET);
        System.out.println(TAG + CYAN + "  XMPP    : " + WHITE + xmppHost + RESET);
        System.out.println(TAG + GREEN + "=============================================" + RESET);

        // TickerBehaviour se declenche toutes les 10 000 ms (10 secondes).
        addBehaviour(new TrendScoutBehaviour(this, 10_000));
    }

    @Override
    protected void takeDown() {
        System.out.println(TAG + RED + "TrendScout agent shutting down" + RESET);
    }

    // ------------------------------------------------------------------
    // Core behaviour
    // ------------------------------------------------------------------

    /**
     * Comportement periodique :
     * 1. GET /api/stats depuis un noeud Flask-Ray
     * 2. Parser genre_votes (compteurs absolus)
     * 3. Si un genre > seuil et different du boost actuel -> broadcast
     */
    private class TrendScoutBehaviour extends TickerBehaviour {

        TrendScoutBehaviour(Agent agent, long period) {
            super(agent, period);
        }

        @Override
        protected void onTick() {
            scanCount++;
            System.out.println();
            System.out.println(TAG + CYAN + "Scanning ecosystem... " + RESET
                    + WHITE + "(scan #" + scanCount + ")" + RESET);

            try {
                String statsJson = fetchStats();
                System.out.println(TAG + BLUE + "Stats received from Flask-Ray cluster" + RESET);

                // On parse les compteurs absolus (genre_votes)
                Map<String, Integer> votes = parseGenreVotes(statsJson);

                if (votes.isEmpty()) {
                    System.out.println(TAG + YELLOW + "No genre data available yet" + RESET);
                    return;
                }

                // Afficher le tableau de bord des votes
                System.out.println(TAG + WHITE + "+--------------------------------------+" + RESET);
                System.out.println(TAG + WHITE + "|  Genre Vote Dashboard                |" + RESET);
                System.out.println(TAG + WHITE + "+--------------------------------------+" + RESET);
                for (Map.Entry<String, Integer> entry : votes.entrySet()) {
                    String bar = "#".repeat(Math.min(entry.getValue(), 20));
                    String color = entry.getValue() > voteThreshold ? GREEN : CYAN;
                    System.out.printf(TAG + "|  " + color + "%-12s %3d " + YELLOW + "%-20s" + RESET + " |%n",
                            entry.getKey(), entry.getValue(), bar);
                }
                System.out.println(TAG + WHITE + "+--------------------------------------+" + RESET);

                String topGenre = analyzeTrends(votes);
                if (topGenre != null) {
                    sendBoost(topGenre);
                } else {
                    System.out.println(TAG + CYAN + "No new trend detected" + RESET);
                }
            } catch (Exception e) {
                System.out.println(TAG + RED + "Tick failed: " + e.getMessage() + RESET);
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 1 : Recuperer les stats de Flask
    // ------------------------------------------------------------------

    private String fetchStats() throws IOException {
        return httpGet(flaskServiceUrl + "/api/stats");
    }

    // ------------------------------------------------------------------
    // Step 2 : Analyser les tendances par genre (seuil en nombre de votes)
    // ------------------------------------------------------------------

    /**
     * Trouve le genre le plus populaire. Le retourne seulement quand il
     * depasse le seuil ET differe du genre actuellement booste.
     */
    private String analyzeTrends(Map<String, Integer> votes) {
        String topGenre = null;
        int topVotes = 0;

        for (Map.Entry<String, Integer> entry : votes.entrySet()) {
            if (entry.getValue() > topVotes) {
                topVotes = entry.getValue();
                topGenre = entry.getKey();
            }
        }

        if (topGenre != null && topVotes > voteThreshold
                && !topGenre.equals(currentBoostedGenre)) {
            System.out.println(TAG + BOLD + YELLOW
                    + ">>> TREND DETECTED: " + topGenre.toUpperCase()
                    + " with " + topVotes + " votes (threshold: >" + voteThreshold + ")"
                    + RESET);
            return topGenre;
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Step 3 : Broadcast BOOST_GENRE via l API HTTP ejabberd
    // ------------------------------------------------------------------

    /**
     * Envoie BOOST_GENRE:<genre> au JID bare du recommender via l endpoint
     * send_message de mod_http_api de ejabberd.
     *
     * Pourquoi l API HTTP plutot qu un client XMPP natif :
     * - Evite d embarquer Smack (~2 MB) aux cotes du transport JADE.
     * - Un seul HTTP POST remplace un cycle de vie complet c2s.
     * - l API ejabberd est deja exposee sur le port 5280 pour le panneau admin.
     */
    private void sendBoost(String genre) {
        try {
            String apiUrl = "http://" + xmppHost + ":5280/api/send_message";
            String body = String.format(
                    "{\"type\":\"chat\","
                            + "\"from\":\"trendscout@kinetoscope.local\","
                            + "\"to\":\"recommender@kinetoscope.local\","
                            + "\"subject\":\"boost\","
                            + "\"body\":\"BOOST_GENRE:%s\"}",
                    genre);

            URL url = new URL(apiUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");

            // Auth Basic requise par api_permissions dans ejabberd.yml
            String credentials = xmppAdminUser + "@kinetoscope.local:" + xmppAdminPass;
            String encoded = Base64.getEncoder().encodeToString(credentials.getBytes());
            conn.setRequestProperty("Authorization", "Basic " + encoded);
            conn.setDoOutput(true);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(5000);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.getBytes());
            }

            int code = conn.getResponseCode();
            if (code == 200) {
                System.out.println(TAG + GREEN + BOLD
                        + "BOOST_GENRE:" + genre + " sent via XMPP -> all Flask replicas"
                        + RESET);
                currentBoostedGenre = genre;
            } else {
                System.out.println(TAG + RED + "XMPP API returned HTTP " + code + RESET);
            }
        } catch (Exception e) {
            System.out.println(TAG + RED + "Failed to send XMPP boost: " + e.getMessage() + RESET);
        }
    }

    // ------------------------------------------------------------------
    // HTTP helper
    // ------------------------------------------------------------------

    private String httpGet(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(5000);
        conn.setReadTimeout(5000);

        int code = conn.getResponseCode();
        if (code != 200) {
            throw new IOException("HTTP " + code + " from " + urlStr);
        }

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream()))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            return sb.toString();
        }
    }

    // ------------------------------------------------------------------
    // Parser JSON leger pour genre_votes (compteurs absolus)
    // ------------------------------------------------------------------

    /**
     * Extrait le map "genre_votes": { ... } du JSON stats
     * sans dependance a une bibliotheque JSON (image Docker minimale).
     *
     * Format attendu :
     * {"genre_votes": {"Action": 3, "Sci-Fi": 7, ...}, ...}
     */
    private Map<String, Integer> parseGenreVotes(String json) {
        Map<String, Integer> result = new HashMap<>();
        try {
            int keyIdx = json.indexOf("\"genre_votes\"");
            if (keyIdx < 0) {
                // Fallback : essayer genre_popularity (fractions) si genre_votes absent
                return parseGenrePopularityLegacy(json);
            }

            int braceStart = json.indexOf("{", keyIdx + 13);
            int braceEnd = json.indexOf("}", braceStart);
            if (braceStart < 0 || braceEnd < 0)
                return result;

            String inner = json.substring(braceStart + 1, braceEnd).trim();
            if (inner.isEmpty())
                return result;

            String[] pairs = inner.split(",");
            for (String pair : pairs) {
                String[] kv = pair.split(":");
                if (kv.length == 2) {
                    String key = kv[0].trim().replace("\"", "");
                    int value = Integer.parseInt(kv[1].trim());
                    result.put(key, value);
                }
            }
        } catch (Exception e) {
            System.out.println(TAG + YELLOW + "JSON parse issue: " + e.getMessage() + RESET);
        }
        return result;
    }

    /**
     * Fallback : parse genre_popularity (fractions) et convertit en pseudo-votes.
     */
    private Map<String, Integer> parseGenrePopularityLegacy(String json) {
        Map<String, Integer> result = new HashMap<>();
        try {
            int keyIdx = json.indexOf("\"genre_popularity\"");
            if (keyIdx < 0)
                return result;

            int braceStart = json.indexOf("{", keyIdx + 18);
            int braceEnd = json.indexOf("}", braceStart);
            if (braceStart < 0 || braceEnd < 0)
                return result;

            String inner = json.substring(braceStart + 1, braceEnd).trim();
            if (inner.isEmpty())
                return result;

            String[] pairs = inner.split(",");
            for (String pair : pairs) {
                String[] kv = pair.split(":");
                if (kv.length == 2) {
                    String key = kv[0].trim().replace("\"", "");
                    double fraction = Double.parseDouble(kv[1].trim());
                    // Convertir fraction en pseudo-votes (estimer sur base 20)
                    result.put(key, (int) Math.round(fraction * 20));
                }
            }
        } catch (Exception e) {
            System.out.println(TAG + YELLOW + "Legacy JSON parse issue: " + e.getMessage() + RESET);
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Env helper
    // ------------------------------------------------------------------

    private static String envOrDefault(String key, String fallback) {
        String val = System.getenv(key);
        return (val != null && !val.isEmpty()) ? val : fallback;
    }
}
