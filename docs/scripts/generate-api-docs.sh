#!/usr/bin/env bash
# Generate API reference markdown from source code
# Run this before building the docs site to update API references

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
OSS_DIR="$(dirname "$DOCS_DIR")"

echo "=== MemoryLayer API Documentation Generator ==="
echo ""

# Generate Python SDK API reference
echo "→ Generating Python SDK API reference..."
if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/extract_python_api.py" \
        "$OSS_DIR/memorylayer-sdk-python/src/memorylayer" \
        "$DOCS_DIR/src/content/docs/sdk-python/api-reference.md" \
        2>/dev/null || echo "  (Python extractor not available, using static docs)"
else
    echo "  (Python not found, skipping)"
fi

# Generate TypeScript SDK API reference
echo "→ Generating TypeScript SDK API reference..."
if command -v npx &>/dev/null && [ -d "$OSS_DIR/memorylayer-sdk-typescript" ]; then
    cd "$OSS_DIR/memorylayer-sdk-typescript"
    npx typedoc --plugin typedoc-plugin-markdown \
        --out "$DOCS_DIR/src/content/docs/sdk-typescript/generated" \
        src/index.ts \
        2>/dev/null || echo "  (TypeDoc not available, using static docs)"
    cd "$DOCS_DIR"
else
    echo "  (npx or SDK not found, skipping)"
fi

# Generate REST API reference from OpenAPI
echo "→ Generating REST API reference from OpenAPI spec..."
if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/extract_openapi.py" \
        "$OSS_DIR/memorylayer-core-python" \
        "$DOCS_DIR/src/content/docs/server/api-reference.md" \
        2>/dev/null || echo "  (OpenAPI extractor not available, using static docs)"
else
    echo "  (Python not found, skipping)"
fi

echo ""
echo "=== Done ==="
echo "Run 'npm run build' to build the documentation site."
