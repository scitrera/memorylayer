#!/usr/bin/env bash
# Local CI — mirrors .github/workflows/ci.yml
# Usage: ./scripts/ci-local.sh [--fix] [component...]
#   --fix       Auto-fix lint/format issues instead of just checking
#   component   One or more of: server, sdk, typescript (default: all)
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

# ──────────────────────────────────────────────────────────────────
# Python server (memorylayer-core-python)
# ──────────────────────────────────────────────────────────────────
run_server() {
    echo -e "\n${YELLOW}━━━ Python: memorylayer-server ━━━${NC}"
    local dir="memorylayer-core-python"

    if [ "$FIX" = true ]; then
        run_step "server: ruff fix"    python3 -m ruff check --fix --unsafe-fixes "$dir"
        run_step "server: ruff format" python3 -m ruff format "$dir"
    else
        run_step "server: ruff check"  python3 -m ruff check "$dir"
        run_step "server: ruff format" python3 -m ruff format --check "$dir"
    fi

    if [ -d "$dir/.venv" ]; then
        (cd "$dir" && source .venv/bin/activate && run_step "server: pytest" python3 -m pytest tests/ -m "not slow and not integration and not llm and not llm_quality" -x -q)
    else
        echo -e "    ${RED}No .venv found in $dir — run: cd $dir && python3 -m venv .venv && pip install -e '.[dev]'${NC}"
        FAILURES+=("server: pytest (no venv)")
    fi
}

# ──────────────────────────────────────────────────────────────────
# Python SDK (memorylayer-sdk-python)
# ──────────────────────────────────────────────────────────────────
run_sdk() {
    echo -e "\n${YELLOW}━━━ Python: memorylayer-client ━━━${NC}"
    local dir="memorylayer-sdk-python"

    if [ "$FIX" = true ]; then
        run_step "sdk: ruff fix"    python3 -m ruff check --fix --unsafe-fixes "$dir"
        run_step "sdk: ruff format" python3 -m ruff format "$dir"
    else
        run_step "sdk: ruff check"  python3 -m ruff check "$dir"
        run_step "sdk: ruff format" python3 -m ruff format --check "$dir"
    fi

    if [ -d "$dir/.venv" ]; then
        (cd "$dir" && source .venv/bin/activate && run_step "sdk: pytest" python3 -m pytest tests/ -x -q)
    else
        echo -e "    ${RED}No .venv found in $dir — run: cd $dir && python3 -m venv .venv && pip install -e '.[dev]'${NC}"
        FAILURES+=("sdk: pytest (no venv)")
    fi
}

# ──────────────────────────────────────────────────────────────────
# TypeScript packages
# ──────────────────────────────────────────────────────────────────
run_typescript() {
    echo -e "\n${YELLOW}━━━ TypeScript: memorylayer-sdk ━━━${NC}"
    (cd memorylayer-sdk-typescript && run_step "ts-sdk: npm ci" npm ci && run_step "ts-sdk: build" npm run build)

    echo -e "\n${YELLOW}━━━ TypeScript: memorylayer-mcp-server ━━━${NC}"
    (cd memorylayer-mcp-typescript && run_step "ts-mcp: npm ci" npm ci && run_step "ts-mcp: build" npm run build)
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
