# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Chỉ giữ system deps tối thiểu (SSL, cert)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Cài uv
RUN pip install --no-cache-dir uv

# Copy dependency files trước để cache tốt
COPY pyproject.toml uv.lock* ./

# Cache uv downloads để build lần sau nhanh hơn
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy source code
COPY src ./src

EXPOSE 8000
WORKDIR /app/src

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t duythong2/virenagents:latest .
#docker build --no-cache -t duythong2/virenagents:latest .
#docker push duythong2/virenagents:latest

