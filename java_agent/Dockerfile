# Multi-stage build for the JADE TrendScout agent.
# Stage 1 (builder): JDK + JADE JAR download + compilation.
# Stage 2 (runtime): JRE-only alpine, ~200 MB total.

# ---- Builder ----
FROM eclipse-temurin:17-jdk-alpine AS builder

WORKDIR /agent

# Download JADE framework JAR from TILAB (official source).
# --no-check-certificate is needed because the tilab.com TLS cert is
# untrusted in Alpine's CA bundle.
RUN apk add --no-cache wget unzip && \
    wget --no-check-certificate -q \
         "https://jade.tilab.com/dl.php?file=JADE-bin-4.6.0.zip" \
         -O jade.zip && \
    unzip -q jade.zip && \
    find . -name "jade.jar" -exec cp {} /agent/jade.jar \; && \
    rm -rf jade.zip jade JADE-bin-4.6.0

COPY src/TrendAgent.java .

# Compile against JADE classpath
RUN javac -cp jade.jar TrendAgent.java

# ---- Runtime ----
FROM eclipse-temurin:17-jre-alpine

WORKDIR /agent

COPY --from=builder /agent/jade.jar .
COPY --from=builder /agent/*.class .

# Start a JADE main container (platform) with the TrendScout agent.
# -gui is omitted — no display available inside Docker.
CMD ["java", "-cp", "jade.jar:.", "jade.Boot", \
     "-agents", "trendscout:TrendAgent"]
