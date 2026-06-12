# Serving image for Hugging Face Spaces (Docker SDK, port 7860)
FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
WORKDIR /home/app/pitchsense

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# config.py creates data/report dirs at import and billing writes
# data/processed/paid_sessions.json, so the app dir must be owned by app
COPY --chown=app:app . .
USER app
ENV HOME=/home/app

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--no-server-header"]
