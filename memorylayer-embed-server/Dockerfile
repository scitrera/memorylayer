# MemoryLayer Embed Server — development image (GPU)
# Build context: repo root (scitrera-memorylayer-ai-cc/)
# docker build -f oss/memorylayer-embed-server/Dockerfile .
#
# This image runs the embed server as a child of the Aether `proxy-sidecar`
# binary in supervisor mode. The sidecar is PID 1, owns the gateway
# connection (sv::memorylayer-embed::*), terminates inbound ProxyHttpRequest
# envelopes, and forwards to the local embed FastAPI on 127.0.0.1:61051.
# The Python embed server itself is plain HTTP and has no Aether knowledge.

# ── Stage 0: build the Go proxy-sidecar binary ───────────────────────
ARG AETHER_REPO_PATH=../scitrera-aether3-go
FROM golang:1.25-bookworm AS sidecar-build

WORKDIR /src

# The build context is the memorylayer repo root; we expect the aether
# repo to be staged into a known location ahead of `docker build`. The
# dev-platform.sh wrapper already does this for ml-server; embed-server
# uses the same staging point for consistency.
COPY proprietary/.build-staging/aether-server /src/aether-server

WORKDIR /src/aether-server/server
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' \
        -o /out/proxy-sidecar ./cmd/proxy-sidecar

# ── Stage 1: compile native Python extensions with full CUDA toolkit ─
FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential curl ca-certificates && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast, torch-backend-aware installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create a venv that we'll copy to the runtime stage
RUN python3 -m venv /opt/memorylayer-embed

# Copy source packages (now all under oss/ — Phase 3 relocation)
COPY oss/memorylayer-core-python /src/memorylayer-core-python
COPY oss/memorylayer-embed-server /src/memorylayer-embed-server
COPY proprietary/.build-staging/aether-python-client /src/aether-python-client

# Install everything into the venv with correct CUDA torch wheels
RUN uv pip install --python /opt/memorylayer-embed/bin/python --torch-backend=auto \
    /src/aether-python-client \
    "/src/memorylayer-core-python[all]" \
    "/src/memorylayer-embed-server[all]"

# ── Stage 2: minimal CUDA runtime ────────────────────────────────────
FROM nvidia/cuda:13.1.1-runtime-ubuntu24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        curl ca-certificates && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 65532 memorylayer && \
    useradd --uid 65532 --gid memorylayer --create-home memorylayer

# Copy the fully-built venv from the builder
COPY --from=builder /opt/memorylayer-embed /opt/memorylayer-embed
ENV PATH="/opt/memorylayer-embed/bin:$PATH" \
    VIRTUAL_ENV="/opt/memorylayer-embed"

# Copy the proxy-sidecar binary (Aether terminator)
COPY --from=sidecar-build /out/proxy-sidecar /usr/local/bin/proxy-sidecar

# Default sidecar config + entrypoint wrapper. The entrypoint script
# branches on EMBED_SERVER_RUN_SIDECAR: default ("true") runs the sidecar
# as PID 1 in supervisor mode; "false" runs the embed server directly as
# plain HTTP (used by docker-compose integration tests and dev shells).
COPY --chown=memorylayer:memorylayer oss/memorylayer-embed-server/configs/proxy-sidecar.default.yaml /etc/aether-sidecar.yaml
COPY --chown=memorylayer:memorylayer --chmod=0755 oss/memorylayer-embed-server/scripts/entrypoint.sh /usr/local/bin/embed-server-entrypoint.sh

WORKDIR /app
USER memorylayer

EXPOSE 61051

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:61051/health || exit 1

ENTRYPOINT ["/usr/local/bin/embed-server-entrypoint.sh"]
