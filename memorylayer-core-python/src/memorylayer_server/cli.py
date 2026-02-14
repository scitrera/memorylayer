"""MemoryLayer.ai CLI - Command line interface for memory infrastructure."""

import click

from scitrera_app_framework import get_variables


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logs")
def cli(verbose: bool):
    """MemoryLayer.ai - Memory infrastructure for LLM-powered agents."""
    v = get_variables()  # get variables instance prior to preconfigure() call
    if verbose:
        v.set("LOGGING_LEVEL", "DEBUG")


@cli.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
def serve(host: str, port: int):
    """Start the HTTP REST API server."""
    import uvicorn
    from memorylayer_server.config import (
        MEMORYLAYER_SERVER_HOST, MEMORYLAYER_SERVER_PORT, DEFAULT_MEMORYLAYER_SERVER_HOST, DEFAULT_MEMORYLAYER_SERVER_PORT
    )
    from memorylayer_server.dependencies import preconfigure
    from memorylayer_server.lifecycle.fastapi import fastapi_app_factory

    # preconfigure ensures that plugins are registered
    v, _ = preconfigure()  # TODO: ideally we would support controlling variables instance?
    if host is None:
        host = v.environ(MEMORYLAYER_SERVER_HOST, default=DEFAULT_MEMORYLAYER_SERVER_HOST)
    if port is None:
        port = v.environ(MEMORYLAYER_SERVER_PORT, default=DEFAULT_MEMORYLAYER_SERVER_PORT, type_fn=int)

    # get FastAPI app instance
    app = fastapi_app_factory(v)

    click.echo(f"Starting memorylayer.ai server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
    )


@cli.command()
def version():
    """Show version information."""
    from memorylayer import __version__
    click.echo(f"memorylayer.ai v{__version__}")


@cli.command()
@click.option('--workspace', '-w', required=True, help='Workspace ID to export')
@click.option('--output', '-o', default=None, help='Output file (default: stdout)')
@click.option('--offset', default=0, type=int, help='Skip first N memories (default: 0)')
@click.option('--limit', default=0, type=int, help='Export at most N memories (default: 0 = unlimited)')
@click.option('--include-associations/--no-associations', default=True, help='Include associations')
@click.option('--server-url', default='http://localhost:61001', help='MemoryLayer server URL')
@click.option('--api-key', default=None, help='API key for authentication')
def export(workspace, output, offset, limit, include_associations, server_url, api_key):
    """Export workspace memories to NDJSON (streaming)."""
    import httpx
    import json

    url = f"{server_url}/v1/workspaces/{workspace}/export"
    params = {
        "offset": offset,
        "limit": limit,
        "include_associations": str(include_associations).lower()
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=300.0) as client:
            with client.stream("GET", url, params=params, headers=headers) as response:
                response.raise_for_status()

                memories_count = 0
                associations_count = 0

                if output:
                    with open(output, 'w') as f:
                        for line in response.iter_lines():
                            if not line.strip():
                                continue
                            f.write(line + '\n')

                            # Parse footer to get counts
                            try:
                                obj = json.loads(line)
                                if obj.get("type") == "footer":
                                    memories_count = obj.get("memories_exported", 0)
                                    associations_count = obj.get("associations_exported", 0)
                            except Exception:
                                pass

                    click.echo(f"Exported {memories_count} memories and {associations_count} associations to {output}")
                else:
                    for line in response.iter_lines():
                        if line.strip():
                            click.echo(line)
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to export workspace: {e}", err=True)
        raise SystemExit(1)


@cli.command(name='import')
@click.argument('file', type=click.Path(exists=True))
@click.option('--workspace', '-w', required=True, help='Target workspace ID')
@click.option('--dry-run', is_flag=True, help='Show what would be imported without writing')
@click.option('--server-url', default='http://localhost:61001', help='MemoryLayer server URL')
@click.option('--api-key', default=None, help='API key for authentication')
def import_cmd(file, workspace, dry_run, server_url, api_key):
    """Import memories from JSON or NDJSON file into workspace."""
    import httpx
    import json

    # Auto-detect format by reading first line
    with open(file, 'r') as f:
        first_line = f.readline().strip()

    is_ndjson = False
    try:
        obj = json.loads(first_line)
        if obj.get("type") == "header":
            is_ndjson = True
    except Exception:
        pass

    if is_ndjson:
        # NDJSON format
        memories = []
        associations = []
        with open(file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj_type = obj.get("type")
                    if obj_type == "memory":
                        memories.append(obj.get("data", {}))
                    elif obj_type == "association":
                        associations.append(obj.get("data", {}))
                except Exception:
                    pass
    else:
        # JSON format
        with open(file, 'r') as f:
            data = json.load(f)
        memories = data.get('memories', [])
        associations = data.get('associations', [])

    if dry_run:
        click.echo(f"Would import {len(memories)} memories and {len(associations)} associations into workspace {workspace}")
        for m in memories[:10]:
            click.echo(f"  - [{m.get('type', 'unknown')}] {m.get('id', 'no-id')}: {(m.get('abstract') or m.get('content', ''))[:80]}")
        if len(memories) > 10:
            click.echo(f"  ... and {len(memories) - 10} more")
        return

    url = f"{server_url}/v1/workspaces/{workspace}/import"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=120.0) as client:
            if is_ndjson:
                # Send as NDJSON
                headers["Content-Type"] = "application/x-ndjson"
                with open(file, 'rb') as f:
                    response = client.post(url, content=f.read(), headers=headers)
            else:
                # Send as JSON
                headers["Content-Type"] = "application/json"
                response = client.post(url, json={"data": {"memories": memories, "associations": associations}}, headers=headers)

            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to import: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Import complete:")
    click.echo(f"  Imported: {result.get('imported', 0)}")
    click.echo(f"  Skipped (duplicates): {result.get('skipped_duplicates', 0)}")
    click.echo(f"  Errors: {result.get('errors', 0)}")
    for detail in result.get('details', []):
        click.echo(f"  {detail}")


@cli.command()
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
def info(output_format: str):
    """Show system information and configuration."""

    from memorylayer_server.dependencies import _initialize_sync
    v = get_variables()
    v.set("LOGGING_LEVEL", "ERROR")  # suppress logs during info output
    v = _initialize_sync(v)
    # TODO: move redaction log to scitrera_app_framework and share with log_framework_variables
    settings = {
        k.removeprefix('MEMORYLAYER_'): '(redacted)' if any(
            (not 'max_tokens' in k.lower()) and x in k.lower() for x in ('password', 'secret', 'credentials', 'token', 'key',)) else val
        for (k, val) in sorted(v.export_all_variables().items(), key=lambda kv: kv[0])
        if k.startswith('MEMORYLAYER')
    }

    if output_format == "json":
        click.echo(json.dumps(settings, indent=2))
    else:
        click.echo("MemoryLayer.ai Configuration")
        click.echo("=" * 40)
        for k, v in settings.items():
            click.echo(f"{k}: {v}")
        click.echo("")


if __name__ == "__main__":
    cli()
