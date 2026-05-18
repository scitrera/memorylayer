#!/usr/bin/env bash
# Local CI — mirrors .github/workflows/ci.yml
# Usage: ./scripts/ci-local.sh [--fix] [component...]
#   --fix       Auto-fix lint/format issues instead of just checking
#   component   One or more of: server, sdk, typescript (default: all)
#
# Python steps shell out to ``uv run --extra dev …`` so each package's
# venv is managed by uv — independent of whatever venv (if any) is
# active in the calling shell. The first invocation per package
# materializes ``<package>/.venv`` with the right extras; subsequent
# runs reuse it.
#
# Examples:
#   ./scripts/ci-local.sh              # Run everything
#   ./scripts/ci-local.sh server       # Server only
#   ./scripts/ci-local.sh sdk --fix    # SDK with auto-fix
#   ./scripts/ci-local.sh server sdk   # Server + SDK, skip TypeScript

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

FIX=false
COMPONENTS=()

for arg in "$@"; do
    case "$arg" in
        --fix) FIX=true ;;
        server|sdk|typescript|ts) COMPONENTS+=("$arg") ;;
        *) echo -e "${RED}Unknown argument: $arg${NC}"; exit 1 ;;
    esac
done

# Default: run all
if [ ${#COMPONENTS[@]} -eq 0 ]; then
    COMPONENTS=(server sdk typescript)
fi

FAILURES=()

run_step() {
    local label="$1"
    shift
    echo -e "${BOLD}  → $label${NC}"
    if "$@"; then
        echo -e "    ${GREEN}✓ passed${NC}"
    else
        echo -e "    ${RED}✗ failed${NC}"
        FAILURES+=("$label")
    fi
}

# Run a step from inside ``$dir`` without spawning a subshell — using a
# subshell breaks failure tracking because mutations to ``FAILURES``
# inside ``(...)`` don't propagate to the parent. pushd/popd keep the
# directory change scoped without losing the array updates.
run_step_in() {
    local dir="$1"
    local label="$2"
    shift 2
    pushd "$dir" >/dev/null
    run_step "$label" "$@"
    popd >/dev/null
}

# Require uv for the Python steps. We intentionally don't fall back to
# raw python/ruff/pytest from PATH — too many subtle "wrong venv" bugs.
need_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo -e "${RED}uv not found on PATH.${NC}"
        echo "Install it with one of:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  brew install uv"
        echo "  pipx install uv"
        exit 1
    fi
}

# ──────────────────────────────────────────────────────────────────
# Python server (memorylayer-core-python)
# ──────────────────────────────────────────────────────────────────
run_server() {
    need_uv
    echo -e "\n${YELLOW}━━━ Python: memorylayer-server ━━━${NC}"
    local dir="memorylayer-core-python"

    # ``uv run`` materializes/refreshes ``$dir/.venv`` with the
    # specified extras before invoking the command. Caller's active
    # venv is irrelevant.
    if [ "$FIX" = true ]; then
        run_step_in "$dir" "server: ruff fix"    uv run --extra dev --extra context -- ruff check --fix --unsafe-fixes .
        run_step_in "$dir" "server: ruff format" uv run --extra dev --extra context -- ruff format .
    else
        run_step_in "$dir" "server: ruff check"  uv run --extra dev --extra context -- ruff check .
        run_step_in "$dir" "server: ruff format" uv run --extra dev --extra context -- ruff format --check .
    fi

    run_step_in "$dir" "server: pytest" uv run --extra dev --extra context -- pytest tests/ -m "not slow and not integration and not llm and not llm_quality" -x -q
}

# ──────────────────────────────────────────────────────────────────
# Python SDK (memorylayer-sdk-python)
# ──────────────────────────────────────────────────────────────────
run_sdk() {
    need_uv
    echo -e "\n${YELLOW}━━━ Python: memorylayer-client ━━━${NC}"
    local dir="memorylayer-sdk-python"

    if [ "$FIX" = true ]; then
        run_step_in "$dir" "sdk: ruff fix"    uv run --extra dev -- ruff check --fix --unsafe-fixes .
        run_step_in "$dir" "sdk: ruff format" uv run --extra dev -- ruff format .
    else
        run_step_in "$dir" "sdk: ruff check"  uv run --extra dev -- ruff check .
        run_step_in "$dir" "sdk: ruff format" uv run --extra dev -- ruff format --check .
    fi

    run_step_in "$dir" "sdk: pytest" uv run --extra dev -- pytest tests/ -x -q
}

# ──────────────────────────────────────────────────────────────────
# TypeScript packages — three-tier dep chain:
#
#   memorylayer-sdk-typescript                (root; no internal deps)
#     └── memorylayer-mcp-typescript          (file:../memorylayer-sdk-typescript)
#           ├── memorylayer-opencode-plugin   (file:../memorylayer-mcp-typescript)
#           └── memorylayer-cc-plugin         (file:../memorylayer-mcp-typescript)
#
# Each tier installs, builds, then runs tests (with --if-present on the
# plugins so missing test scripts skip cleanly).
# ──────────────────────────────────────────────────────────────────
run_typescript() {
    echo -e "\n${YELLOW}━━━ TypeScript: memorylayer-sdk ━━━${NC}"
    run_step_in memorylayer-sdk-typescript "ts-sdk: npm ci" npm ci
    run_step_in memorylayer-sdk-typescript "ts-sdk: build"  npm run build
    run_step_in memorylayer-sdk-typescript "ts-sdk: test"   npm test

    echo -e "\n${YELLOW}━━━ TypeScript: memorylayer-mcp-server ━━━${NC}"
    run_step_in memorylayer-mcp-typescript "ts-mcp: npm ci" npm ci
    run_step_in memorylayer-mcp-typescript "ts-mcp: build"  npm run build
    run_step_in memorylayer-mcp-typescript "ts-mcp: test"   npm test

    # Plugins depend on MCP being built (file:../memorylayer-mcp-typescript).
    # MCP was just built above, so npm ci here picks up its dist/.
    for plugin in memorylayer-opencode-plugin memorylayer-cc-plugin; do
        echo -e "\n${YELLOW}━━━ TypeScript: ${plugin} ━━━${NC}"
        run_step_in "$plugin" "${plugin}: npm ci" npm ci
        run_step_in "$plugin" "${plugin}: build"  npm run build
        run_step_in "$plugin" "${plugin}: test"   npm test --if-present
    done
}

# ──────────────────────────────────────────────────────────────────
# Run selected components
# ──────────────────────────────────────────────────────────────────
for component in "${COMPONENTS[@]}"; do
    case "$component" in
        server)     run_server ;;
        sdk)        run_sdk ;;
        typescript|ts) run_typescript ;;
    esac
done

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
echo ""
if [ ${#FAILURES[@]} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}${#FAILURES[@]} check(s) failed:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "  ${RED}✗ $f${NC}"
    done
    exit 1
fi
