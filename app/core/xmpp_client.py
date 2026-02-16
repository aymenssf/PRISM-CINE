"""
PRISM CINE — Client XMPP reliant les messages ejabberd au Ray actor.

Ici on utilise XMPP (plutot qu un callback REST) car :
    - Livraison de messages en temps-reel sans polling.
    - ejabberd multicast vers TOUTES les ressources connectees du JID bare,
      donc les deux replicas Flask recoivent BOOST_GENRE en un seul envoi.
    - Les replicas Flask n ont pas besoin d exposer un webhook que l agent
      JADE devrait decouvrir derriere des ports dynamiques.
"""

import slixmpp
import logging
import ray
from typing import Any, Optional, Callable

logger = logging.getLogger("prismcine.xmpp")


class PRISMCINEXMPPClient(slixmpp.ClientXMPP):
    """
    Se connecte a ejabberd en tant que ``recommender@prismcine.local/<hostname>``.

    Ecoute deux types de messages :
        UPDATE              – acquittement seul (futur : declencher recalcul)
        BOOST_GENRE:<genre> – transmis au Ray actor + callback Flask
    """

    def __init__(
        self,
        jid: str,
        password: str,
        recommender_actor: Any,
        boost_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(jid, password)
        self.recommender_actor = recommender_actor
        self.boost_callback = boost_callback

        # Plugins : decouverte de services + keepalive ping
        self.register_plugin("xep_0030")  # Service Discovery
        self.register_plugin("xep_0199")  # XMPP Ping

        self.add_event_handler("session_start", self.on_session_start)
        self.add_event_handler("message", self.on_message)
        self.add_event_handler("disconnected", self.on_disconnected)

        print(f"[XMPP-BUS] Client created for JID: {jid}")
        logger.info("[XMPP-BUS] XMPP client created for JID: %s", jid)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def on_session_start(self, _event):
        """Envoie la presence initiale et recupere le roster apres auth."""
        try:
            self.send_presence()
            await self.get_roster()
            print("[XMPP-BUS] Session started -- presence sent")
            logger.info("[XMPP-BUS] XMPP session started -- presence sent")
        except Exception as exc:
            print(f"[XMPP-BUS] Failed to start session: {exc}")
            logger.error("[XMPP-BUS] Failed to start XMPP session: %s", exc)

    def on_message(self, msg):
        """
        Route les messages entrants vers le handler approprie.

        Contrat de protocole (defini entre l agent JADE et les noeuds Flask) :
            "UPDATE"               -> log d acquittement
            "BOOST_GENRE:<genre>"  -> forward genre au Ray actor + callback Flask
        """
        if msg["type"] not in ("chat", "normal"):
            return

        body = msg["body"].strip()
        sender = msg["from"]
        print(f"[XMPP-BUS] Message from {sender}: {body}")
        logger.info("[XMPP-BUS] XMPP message from %s: %s", sender, body)

        try:
            if body == "UPDATE":
                print("[XMPP-BUS] UPDATE command acknowledged")
                logger.info("[XMPP-BUS] UPDATE command acknowledged")

            elif body.startswith("BOOST_GENRE:"):
                genre = body.split(":", 1)[1].strip()
                if not genre:
                    print("[XMPP-BUS] WARNING: Empty genre in BOOST_GENRE message")
                    logger.warning("[XMPP-BUS] Empty genre in BOOST_GENRE message")
                    return

                print(f"[XMPP-BUS] Received Boost Order! Genre = {genre}")
                logger.info("[XMPP-BUS] Received Boost Order! Genre=%s", genre)

                # ray.get() bloque mais l agent JADE envoie au maximum toutes les 10s,
                # et boost_genre() est une simple assignation de champ (microsecondes).
                ray.get(self.recommender_actor.boost_genre.remote(genre))
                print(f"[XMPP-BUS] Genre boost applied via XMPP: {genre}")
                logger.info("[XMPP-BUS] Genre boost applied via XMPP: %s", genre)

                # Callback vers Flask pour mettre a jour la variable globale BOOSTED_GENRE
                if self.boost_callback:
                    self.boost_callback(genre)
                    print(f"[XMPP-BUS] Flask callback executed for genre: {genre}")

            else:
                logger.debug("[XMPP-BUS] Unrecognised XMPP message: %s", body)

        except Exception as exc:
            print(f"[XMPP-BUS] Error processing message '{body}': {exc}")
            logger.error("[XMPP-BUS] Error processing XMPP message '%s': %s", body, exc)

    def on_disconnected(self, _event):
        """Log la deconnexion ; slixmpp tentera la reconnexion automatique."""
        print("[XMPP-BUS] Disconnected -- reconnection will be attempted")
        logger.warning("[XMPP-BUS] XMPP disconnected -- reconnection will be attempted")
