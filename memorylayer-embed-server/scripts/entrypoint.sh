#!/bin/sh
# Embed-server container entrypoint.
#
# Two modes:
#   * Default (EMBED_SERVER_RUN_SIDECAR unset or "true"):
#       proxy-sidecar runs as PID 1, supervises the embed FastAPI on
#       127.0.0.1:61051, and exposes the Aether terminator surface
#       (sv::memorylayer-embed::*) to the gateway.
#
#   * Plain HTTP (EMBED_SERVER_RUN_SIDECAR=false):
#       memorylayer-embed serve runs as PID 1. No Aether. Suitable for
#       docker-compose / CI integration tests where the embed-server is
#       reached via direct HTTP on port 61051.
set -e

if [ "${EMBED_SERVER_RUN_SIDECAR:-true}" = "false" ]; then
    echo "embed-server entrypoint: plain HTTP mode (no sidecar)" >&2
    exec memorylayer-embed serve "$@"
fi

CONFIG_PATH="${EMBED_SERVER_SIDECAR_CONFIG:-/etc/aether-sidecar.yaml}"
echo "embed-server entrypoint: sidecar mode (config=${CONFIG_PATH})" >&2
exec /usr/local/bin/proxy-sidecar --config "${CONFIG_PATH}" -- memorylayer-embed serve "$@"
