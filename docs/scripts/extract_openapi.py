#!/usr/bin/env python3
"""Extract REST API documentation from FastAPI OpenAPI schema.

Starts the FastAPI app, fetches the OpenAPI JSON, and generates
Starlight-compatible markdown documentation.

Usage:
    python extract_openapi.py <server_dir> <output_file>

Falls back gracefully if the server package is not installed.
"""

import json
import sys
from pathlib import Path


def format_endpoint(path: str, method: str, spec: dict) -> str:
    """Format a single API endpoint as markdown."""
    lines = []
    summary = spec.get("summary", "")
    description = spec.get("description", "")

    lines.append(f"### {method.upper()} {path}")
    lines.append("")

    if summary:
        lines.append(summary)
        lines.append("")

    if description and description != summary:
        lines.append(description)
        lines.append("")

    # Parameters
    params = spec.get("parameters", [])
    if params:
        lines.append("**Parameters:**")
        lines.append("")
        lines.append("| Name | In | Type | Required | Description |")
        lines.append("|------|-----|------|----------|-------------|")
        for p in params:
            schema = p.get("schema", {})
            ptype = schema.get("type", "string")
            required = "Yes" if p.get("required") else "No"
            desc = p.get("description", "")
            lines.append(f"| `{p['name']}` | {p['in']} | {ptype} | {required} | {desc} |")
        lines.append("")

    # Request body
    body = spec.get("requestBody", {})
    if body:
        content = body.get("content", {})
        json_schema = content.get("application/json", {}).get("schema", {})
        if json_schema:
            lines.append("**Request Body:** `application/json`")
            lines.append("")
            props = json_schema.get("properties", {})
            required_fields = json_schema.get("required", [])
            if props:
                lines.append("| Field | Type | Required | Description |")
                lines.append("|-------|------|----------|-------------|")
                for name, prop in props.items():
                    ptype = prop.get("type", "any")
                    req = "Yes" if name in required_fields else "No"
                    desc = prop.get("description", "")
                    lines.append(f"| `{name}` | {ptype} | {req} | {desc} |")
                lines.append("")

    # Responses
    responses = spec.get("responses", {})
    if responses:
        for code, resp in responses.items():
            lines.append(f"**Response:** `{code}` {resp.get('description', '')}")
            lines.append("")

    return "\n".join(lines)


def generate_from_openapi(openapi: dict, output_file: Path) -> None:
    """Generate markdown from OpenAPI spec."""
    info = openapi.get("info", {})
    paths = openapi.get("paths", {})

    lines = [
        "---",
        "title: REST API Reference",
        "description: Auto-generated REST API reference from OpenAPI specification",
        "---",
        "",
        ":::note",
        "This reference is auto-generated from the server's OpenAPI schema.",
        ":::",
        "",
        f"**Version:** {info.get('version', 'unknown')}",
        "",
    ]

    # Group by tag
    tagged: dict[str, list[str]] = {}
    for path, methods in sorted(paths.items()):
        for method, spec in methods.items():
            if method in ("get", "post", "put", "delete", "patch"):
                tags = spec.get("tags", ["Other"])
                tag = tags[0] if tags else "Other"
                if tag not in tagged:
                    tagged[tag] = []
                tagged[tag].append(format_endpoint(path, method, spec))

    for tag, endpoints in tagged.items():
        lines.append(f"## {tag}")
        lines.append("")
        for ep in endpoints:
            lines.append(ep)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))
    print(f"Generated: {output_file}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <server_dir> <output_file>")
        sys.exit(1)

    server_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Try to import and get OpenAPI spec
    try:
        sys.path.insert(0, str(server_dir / "src"))
        from memorylayer_server.api.app import create_app
        app = create_app()
        openapi = app.openapi()
        generate_from_openapi(openapi, output_file)
    except ImportError:
        # Try loading from a cached openapi.json
        cached = server_dir / "openapi.json"
        if cached.exists():
            with open(cached) as f:
                openapi = json.load(f)
            generate_from_openapi(openapi, output_file)
        else:
            print("Server package not installed and no cached openapi.json found.")
            print("Install with: pip install -e oss/memorylayer-core-python")
            print("Using static API reference instead.")
            sys.exit(0)


if __name__ == "__main__":
    main()
