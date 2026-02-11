FROM python:3.12-slim AS base

LABEL maintainer="Scitrera <open-source-team@scitrera.com>"
LABEL org.opencontainers.image.source="https://github.com/scitrera/memorylayer"
LABEL org.opencontainers.image.description="MemoryLayer.ai - Memory infrastructure for LLM-powered agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by native extensions (sqlite-vec, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 memorylayer && \
    useradd --uid 1000 --gid memorylayer --create-home memorylayer

# Set up persistent data directory
RUN mkdir -p /data && chown memorylayer:memorylayer /data
VOLUME /data

WORKDIR /app

# Install memorylayer-server with all optional dependencies from PyPI
RUN pip install --no-cache-dir "memorylayer-server[all]"

# Remove build tools no longer needed at runtime
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER memorylayer

# Container defaults:
# - Bind to 0.0.0.0 (code default 127.0.0.1 is unreachable from outside the container)
# - Use local sentence-transformers for embeddings (no API key required)
ENV MEMORYLAYER_SERVER_HOST=0.0.0.0 \
    MEMORYLAYER_SERVER_PORT=61001 \
    MEMORYLAYER_DATA_DIR=/data \
    MEMORYLAYER_EMBEDDING_PROVIDER=local

EXPOSE 61001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:61001/health || exit 1

ENTRYPOINT ["memorylayer"]
CMD ["serve"]
