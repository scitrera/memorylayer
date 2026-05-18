# MemoryLayer Embed Server — GPU image
# Build context: ``oss/`` (the OSS repo root, or the monorepo's ``oss/`` subdir).
#
#   CI:    docker build -f memorylayer-embed-server/Dockerfile .       # from oss/
#   Dev:   docker build -f oss/memorylayer-embed-server/Dockerfile oss/
#
# The image runs the embed server as a child of the Aether ``proxy-sidecar``
# binary in supervisor mode. The sidecar is PID 1, owns the gateway
# connection (sv::memorylayer-embed::*), terminates inbound ProxyHttpRequest
# envelopes, and forwards to the local embed FastAPI on 127.0.0.1:61051.
# The Python embed server itself is plain HTTP and has no Aether knowledge.
#
# Aether dependencies are pulled from upstream releases — no local Go build:
#   * proxy-sidecar binary  → ghcr.io/scitrera/aether:${AETHER_VERSION}
#   * scitrera-aether-client (Python) → PyPI, optional ${AETHER_CLIENT_VERSION} pin

# ── Stage 0: pull the prebuilt proxy-sidecar binary from upstream ────
# AETHER_VERSION accepts any tag published by ghcr.io/scitrera/aether
# (no leading "v"). Default ``latest`` pins to the most recent release;
# CI builds may override via build-args to lock a known-good revision.
ARG AETHER_VERSION=latest
FROM ghcr.io/scitrera/aether:${AETHER_VERSION} AS aether-image

# ── Stage 1: compile native Python extensions with full CUDA toolkit ─
FROM nvidia/cuda:13.1.2-devel-ubuntu24.04 AS builder

# Empty AETHER_CLIENT_VERSION installs the latest scitrera-aether-client
# from PyPI; supply a concrete version (e.g. ``0.2.0``) to pin.
ARG AETHER_CLIENT_VERSION=""

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

# Copy source packages (relative to the oss/ build context).
COPY memorylayer-core-python /src/memorylayer-core-python
COPY memorylayer-embed-server /src/memorylayer-embed-server

# Install everything into the venv with correct CUDA torch wheels.
# scitrera-aether-client comes from PyPI: empty AETHER_CLIENT_VERSION
# → latest; otherwise pinned to the supplied version.
RUN set -eu; \
    if [ -n "${AETHER_CLIENT_VERSION}" ]; then \
        AETHER_PKG="scitrera-aether-client==${AETHER_CLIENT_VERSION}"; \
    else \
        AETHER_PKG="scitrera-aether-client"; \
    fi; \
    uv pip install --python /opt/memorylayer-embed/bin/python --torch-backend=auto \
        "${AETHER_PKG}" \
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

# Copy the prebuilt proxy-sidecar binary from the upstream Aether image
COPY --from=aether-image /usr/local/bin/proxy-sidecar /usr/local/bin/proxy-sidecar

# Default sidecar config + entrypoint wrapper. The entrypoint script
# branches on EMBED_SERVER_RUN_SIDECAR: default ("true") runs the sidecar
# as PID 1 in supervisor mode; "false" runs the embed server directly as
# plain HTTP (used by docker-compose integration tests and dev shells).
COPY --chown=memorylayer:memorylayer memorylayer-embed-server/configs/proxy-sidecar.default.yaml /etc/aether-sidecar.yaml
COPY --chown=memorylayer:memorylayer --chmod=0755 memorylayer-embed-server/scripts/entrypoint.sh /usr/local/bin/embed-server-entrypoint.sh

WORKDIR /app
USER memorylayer

EXPOSE 61051

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:61051/health || exit 1

ENTRYPOINT ["/usr/local/bin/embed-server-entrypoint.sh"]
