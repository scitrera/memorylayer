#!/usr/bin/env bash
# Run ``npm audit`` (and optionally ``npm audit fix``) across every
# TypeScript / npm package in this repo.
#
# Usage: ./scripts/npm-audit.sh [--fix] [--force] [--level <lvl>] [pkg...]
#   --fix          Run ``npm audit fix`` before the audit reporting step.
#   --force        Pass ``--force`` to ``npm audit fix`` (breaking changes).
#                  Only meaningful with ``--fix``.
#   --level <lvl>  Only report vulnerabilities at this severity or higher.
#                  One of: info | low | moderate | high | critical.
#   pkg…           Audit a subset. Either full package dir name or alias:
#                  ``sdk``, ``mcp``, ``opencode``, ``cc``.
#
# Examples:
#   ./scripts/npm-audit.sh                          # Audit all four packages
#   ./scripts/npm-audit.sh --fix                    # Audit + non-breaking fix
#   ./scripts/npm-audit.sh --fix --force            # Including breaking fixes
#   ./scripts/npm-audit.sh --level high             # Only high + critical
#   ./scripts/npm-audit.sh sdk mcp                  # Just sdk + mcp
#
# Exit codes:
#   0  All packages clean at the configured audit level
#   1  At least one ``npm audit`` reported issues

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Source-of-truth list of TS packages to audit. Add new ones here.
TS_PACKAGES=(
    memorylayer-sdk-typescript
    memorylayer-mcp-typescript
    memorylayer-opencode-plugin
    memorylayer-cc-plugin
)

# Short alias → full directory name. Keeps invocation friendly.
declare -A TS_ALIAS=(
    [sdk]=memorylayer-sdk-typescript
    [mcp]=memorylayer-mcp-typescript
    [opencode]=memorylayer-opencode-plugin
    [cc]=memorylayer-cc-plugin
)

FIX=false
FORCE=false
LEVEL=""
SELECTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --fix)   FIX=true ;;
        --force) FORCE=true ;;
        --level)
            shift
            LEVEL="${1:-}"
            if [ -z "$LEVEL" ]; then
                echo -e "${RED}--level requires an argument${NC}"; exit 1
            fi
            ;;
        --level=*) LEVEL="${1#--level=}" ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        memorylayer-*) SELECTED+=("$1") ;;
        *)
            if [ -n "${TS_ALIAS[$1]:-}" ]; then
                SELECTED+=("${TS_ALIAS[$1]}")
            else
                echo -e "${RED}Unknown argument: $1${NC}"
                echo "Valid packages: ${TS_PACKAGES[*]}"
                echo "Aliases:        sdk mcp opencode cc"
                exit 1
            fi
            ;;
    esac
    shift
done

if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("${TS_PACKAGES[@]}")
fi

if [ "$FORCE" = true ] && [ "$FIX" = false ]; then
    echo -e "${YELLOW}--force has no effect without --fix; ignoring.${NC}"
fi

# Compose the level flag once so audit + audit-fix both honor it.
LEVEL_ARGS=()
if [ -n "$LEVEL" ]; then
    LEVEL_ARGS=(--audit-level "$LEVEL")
fi

FAILURES=()

run_step() {
    local label="$1"
    shift
    echo -e "${BOLD}  → $label${NC}"
    if "$@"; then
        echo -e "    ${GREEN}✓ clean${NC}"
    else
        echo -e "    ${RED}✗ issues${NC}"
        FAILURES+=("$label")
    fi
}

# pushd/popd (no subshell) so FAILURES mutations survive — same pattern as
# ci-local.sh after fixing its silent-green bug.
run_step_in() {
    local dir="$1"
    local label="$2"
    shift 2
    pushd "$dir" >/dev/null
    run_step "$label" "$@"
    popd >/dev/null
}

# ``npm audit`` needs ``package-lock.json`` and a populated node_modules to
# read the dep tree. If either is missing, run ``npm ci`` first.
ensure_installed() {
    local dir="$1"
    if [ ! -f "$dir/package-lock.json" ]; then
        echo -e "  ${RED}$dir has no package-lock.json — skipping audit${NC}"
        FAILURES+=("$dir: missing lockfile")
        return 1
    fi
    if [ ! -d "$dir/node_modules" ]; then
        echo -e "  ${YELLOW}↻ priming node_modules in $dir (npm ci)${NC}"
        pushd "$dir" >/dev/null
        npm ci --no-audit --no-fund --silent
        popd >/dev/null
    fi
    return 0
}

for pkg in "${SELECTED[@]}"; do
    echo -e "\n${YELLOW}━━━ npm-audit: $pkg ━━━${NC}"
    if ! ensure_installed "$pkg"; then
        continue
    fi

    if [ "$FIX" = true ]; then
        if [ "$FORCE" = true ]; then
            run_step_in "$pkg" "$pkg: audit fix --force" npm audit fix --force "${LEVEL_ARGS[@]}"
        else
            run_step_in "$pkg" "$pkg: audit fix" npm audit fix "${LEVEL_ARGS[@]}"
        fi
    fi

    # Always end with a plain audit so the summary reflects the
    # post-fix state of each package.
    run_step_in "$pkg" "$pkg: audit" npm audit "${LEVEL_ARGS[@]}"
done

echo ""
if [ ${#FAILURES[@]} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}No outstanding vulnerabilities.${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}${#FAILURES[@]} audit step(s) reported issues:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "  ${RED}✗ $f${NC}"
    done
    exit 1
fi
