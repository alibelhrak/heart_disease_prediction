FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy project
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/processed/scaler.joblib ./data/processed/scaler.joblib
COPY data/processed/feature_names.json ./data/processed/feature_names.json
COPY start.sh ./start.sh

RUN chmod +x start.sh

# Optional: keep non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PATH="/usr/local/bin:$PATH"

EXPOSE 8000 5000

CMD ["bash", "start.sh"]