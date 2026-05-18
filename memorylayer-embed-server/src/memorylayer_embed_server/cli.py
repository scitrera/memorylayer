"""MemoryLayer Embed Server CLI."""

import click

from scitrera_app_framework import get_variables


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logs")
def cli(verbose: bool):
    """MemoryLayer Embed Server - Stateless transcription and embedding GPU server."""
    v = get_variables()
    if verbose:
        v.set("LOGGING_LEVEL", "DEBUG")


@cli.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
def serve(host: str, port: int):
    """Start the embed server."""
    import uvicorn
    from memorylayer_embed_server.config import (
        EMBED_SERVER_HOST, EMBED_SERVER_PORT,
        DEFAULT_EMBED_SERVER_HOST, DEFAULT_EMBED_SERVER_PORT,
    )
    from memorylayer_embed_server.dependencies import preconfigure
    from memorylayer_embed_server.lifecycle.fastapi import fastapi_app_factory

    v, _ = preconfigure()
    if host is None:
        host = v.environ(EMBED_SERVER_HOST, default=DEFAULT_EMBED_SERVER_HOST)
    if port is None:
        port = v.environ(EMBED_SERVER_PORT, default=DEFAULT_EMBED_SERVER_PORT, type_fn=int)

    app = fastapi_app_factory(v)

    click.echo(f"Starting MemoryLayer Embed Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)


@cli.command()
def version():
    """Show version information."""
    from memorylayer_embed_server import __version__
    click.echo(f"memorylayer-embed-server v{__version__}")


if __name__ == "__main__":
    cli()
