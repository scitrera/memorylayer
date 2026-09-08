FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native build dependencies belong only in this disposable stage. The virtual
# environment is created at its final runtime path so console-script shebangs
# remain valid after it is copied into the runtime image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /opt/memorylayer

# Build from the SOURCE TREE, not from PyPI. The root .dockerignore excludes
# local virtualenvs, .slop, caches, and agent state before the context reaches
# the Docker daemon; this stage is discarded after installation as a second
# boundary preventing source/build artifacts from entering the runtime image.
COPY memorylayer-core-python /src/memorylayer-core-python
COPY memorylayer-server-rpg-python /src/memorylayer-server-rpg-python
RUN /opt/memorylayer/bin/pip install --no-cache-dir \
    "/src/memorylayer-core-python[all]" \
    /src/memorylayer-server-rpg-python


FROM python:3.12-slim AS runtime

LABEL maintainer="scitrera.ai <open-source-team@scitrera.com>"
LABEL org.opencontainers.image.source="https://github.com/scitrera/memorylayer"
LABEL org.opencontainers.image.description="MemoryLayer.ai - Memory infrastructure for LLM-powered agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/memorylayer/bin:$PATH" \
    VIRTUAL_ENV=/opt/memorylayer

# curl is used by the container healthcheck. Compiler/build packages remain in
# the builder stage and are absent from the shipped runtime filesystem.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 memorylayer && \
    useradd --uid 1000 --gid memorylayer --create-home memorylayer

# Set up persistent data directory
RUN mkdir -p /data && chown memorylayer:memorylayer /data
VOLUME /data

WORKDIR /app

# Copy only the installed runtime environment. The package is still built from
# the checked-out source, while the source tree and compiler toolchain never
# become layers in the final image.
COPY --from=builder /opt/memorylayer /opt/memorylayer

# Switch to non-root user
USER memorylayer

# Container defaults:
# - Bind to 0.0.0.0 to be accessible from outside the container
# - Use port 61001, the default for memorylayer-server
# - Pin the embedding provider to embed_server. The library default is 'hash'
#   (lexical, dependency-free) so that a bare `pip install` works offline, but
#   this image ships the full extras and is meant to run against a
#   memorylayer-embed-server peer. Pinning it here keeps that behavior explicit
#   and makes a missing peer a loud startup error rather than a silent downgrade
#   to lexical matching. Override with -e MEMORYLAYER_EMBEDDING_PROVIDER=openai
#   (or google/hash) to use a different one — see the README.
ENV MEMORYLAYER_SERVER_HOST=0.0.0.0 \
    MEMORYLAYER_SERVER_PORT=61001 \
    MEMORYLAYER_EMBEDDING_PROVIDER=embed_server \
    MEMORYLAYER_DATA_DIR=/data

EXPOSE 61001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:61001/health || exit 1

ENTRYPOINT ["memorylayer"]
CMD ["serve"]
