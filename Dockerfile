# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (Docling + Torch + OpenCV related)
# Cache mount keeps apt lists + downloaded debs between builds — not baked into the image layer
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Install dependencies
# Cache mount keeps downloaded wheels between builds — not baked into the image layer
# This layer is only invalidated when requirements.txt changes
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy source with correct ownership in a single layer (avoids a redundant chown layer)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
