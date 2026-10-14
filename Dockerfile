# API image for one model version. Build from the repo root:
#   docker build -t corvita-icu:v1 .
#   docker run --rm -p 8000:8000 -e LOG_PATH=/app/logs/requests.jsonl -v "$PWD/logs:/app/logs" corvita-icu:v1
#
# Python patch version matches the one that trained models/v1 (see metadata.json -> versions).
FROM python:3.13.1-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/v1

WORKDIR /app

# Exact versions of every package, including indirect ones.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY config.yaml .
COPY src/ src/
COPY api/ api/
COPY models/v1/model.joblib models/v1/metadata.json models/v1/

# The app runs as a non-root user. Give it a writable log folder; without this,
# LOG_PATH=/app/logs/... fails with PermissionError.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/logs \
 && chown app:app /app/logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --no-access-log: the app writes its own JSON line per request. Uvicorn's plain-text
# access log would mix non-JSON lines into stdout and duplicate every request.
CMD ["uvicorn", "api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
