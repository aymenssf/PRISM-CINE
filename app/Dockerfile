# Python service image — Flask + Ray + Slixmpp + NumPy
# Uses python:3.10-slim to keep image size manageable (~150 MB base)
# while still providing pip and C compilation headers for Ray/numpy.

FROM python:3.10-slim

# gcc is required for compiling C extensions used by Ray and numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
