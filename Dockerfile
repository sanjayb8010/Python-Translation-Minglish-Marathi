# Translation API (Flask) — production-style run with gunicorn
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
COPY templates/ templates/

USER appuser

EXPOSE 5000

# Bind 0.0.0.0 so the server is reachable from outside the container.
# Workers: override with docker run ... gunicorn ... if you need a different count.
CMD exec gunicorn --bind "0.0.0.0:${PORT}" --workers 2 app:app
