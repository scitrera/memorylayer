"""MemoryLayer.ai CLI - Command line interface for memory infrastructure."""

import json
import logging
from datetime import UTC

import click
from scitrera_app_framework import get_variables

logger = logging.getLogger(__name__)


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
        DEFAULT_MEMORYLAYER_SERVER_HOST,
        DEFAULT_MEMORYLAYER_SERVER_PORT,
        MEMORYLAYER_SERVER_HOST,
        MEMORYLAYER_SERVER_PORT,
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
    from memorylayer_server import __version__

    click.echo(f"memorylayer.ai v{__version__}")


@cli.command()
@click.option("--workspace", "-w", required=True, help="Workspace ID to export")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option("--offset", default=0, type=int, help="Skip first N memories (default: 0)")
@click.option("--limit", default=0, type=int, help="Export at most N memories (default: 0 = unlimited)")
@click.option("--include-associations/--no-associations", default=True, help="Include associations")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def export(workspace, output, offset, limit, include_associations, server_url, api_key):
    """Export workspace memories to NDJSON (streaming)."""
    import json

    import httpx

    url = f"{server_url}/v1/workspaces/{workspace}/export"
    params = {"offset": offset, "limit": limit, "include_associations": str(include_associations).lower()}
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
                    with open(output, "w") as f:
                        for line in response.iter_lines():
                            if not line.strip():
                                continue
                            f.write(line + "\n")

                            # Parse footer to get counts
                            try:
                                obj = json.loads(line)
                                if obj.get("type") == "footer":
                                    memories_count = obj.get("memories_exported", 0)
                                    associations_count = obj.get("associations_exported", 0)
                            except Exception as e:
                                logger.debug("Skipped line during export footer parse: %s", e)

                    click.echo(f"Exported {memories_count} memories and {associations_count} associations to {output}")
                else:
                    for line in response.iter_lines():
                        if line.strip():
                            click.echo(line)
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to export workspace: {e}", err=True)
        raise SystemExit(1)


@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--workspace", "-w", required=True, help="Target workspace ID")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without writing")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def import_cmd(file, workspace, dry_run, server_url, api_key):
    """Import memories from JSON or NDJSON file into workspace."""
    import json

    import httpx

    # Auto-detect format by reading first line
    with open(file) as f:
        first_line = f.readline().strip()

    is_ndjson = False
    try:
        obj = json.loads(first_line)
        if obj.get("type") == "header":
            is_ndjson = True
    except Exception as e:
        logger.debug("Skipped item during processing: %s", e)

    if is_ndjson:
        # NDJSON format
        memories = []
        associations = []
        with open(file) as f:
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
                except Exception as e:
                    logger.debug("Skipped item during processing: %s", e)
    else:
        # JSON format
        with open(file) as f:
            data = json.load(f)
        memories = data.get("memories", [])
        associations = data.get("associations", [])

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
                with open(file, "rb") as f:
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

    click.echo("Import complete:")
    click.echo(f"  Imported: {result.get('imported', 0)}")
    click.echo(f"  Skipped (duplicates): {result.get('skipped_duplicates', 0)}")
    click.echo(f"  Errors: {result.get('errors', 0)}")
    for detail in result.get("details", []):
        click.echo(f"  {detail}")


@cli.command()
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option("--reveal-secrets", is_flag=True, help="Reveal secret keys in text output")
def info(output_format: str, reveal_secrets: bool):
    """Show system information and configuration."""

    from datetime import datetime

    from memorylayer_server import __version__
    from memorylayer_server.dependencies import _initialize_sync

    v = get_variables()
    v.set("LOGGING_LEVEL", "ERROR")  # suppress logs during info output
    v = _initialize_sync(v)

    redact_keys: bool = not reveal_secrets

    # TODO: move redaction log to scitrera_app_framework and share with log_framework_variables
    def _redacted(k, val):
        return (
            "(redacted)"
            if any(
                ("max_tokens" not in k.lower()) and x in k.lower()
                for x in (
                    "password",
                    "secret",
                    "credentials",
                    "token",
                    "key",
                )
            )
            else val
        )

    settings = (
        {k: _redacted(k, v) for (k, v) in sorted(v.export_all_variables().items(), key=lambda kv: kv[0]) if k.startswith("MEMORYLAYER")}
        if redact_keys
        else {k: v for k, v in sorted(v.export_all_variables().items(), key=lambda kv: kv[0]) if k.startswith("MEMORYLAYER")}
    )

    if output_format == "json":
        click.echo(json.dumps({k.removeprefix("MEMORYLAYER_").lower(): v for k, v in settings.items()}, indent=2))
    else:
        click.echo("# " + "=" * 50)
        click.echo("# MemoryLayer.ai Configuration")
        click.echo(f"# exported at {datetime.now(tz=UTC).isoformat()}")
        click.echo(f"# version = v{__version__}")
        click.echo("# " + "=" * 50)
        for k, v in settings.items():
            click.echo(f"{k}={v}")
        click.echo("")


@cli.group()
def skills():
    """Manage agent skills stored in MemoryLayer."""


@skills.command(name="list")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--name", "-n", default=None, help="Filter by skill name")
@click.option("--show-shadowed", is_flag=True, help="Show all skills including shadowed duplicates")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def skills_list(workspace, name, show_shadowed, server_url, api_key):
    """List visible skills."""
    import httpx

    params = {}
    if workspace:
        params["workspace_id"] = workspace
    if name:
        params["name"] = name
    if show_shadowed:
        params["include_shadowed"] = "true"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{server_url}/v1/skills", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    skill_list = data.get("skills", [])
    if not skill_list:
        click.echo("No skills found.")
        return

    for s in skill_list:
        scope = "user" if s.get("user_id") else "workspace"
        click.echo(f"  [{s.get('version', '?')}] {s['name']} ({scope}) — {s.get('description', '')[:60]}")
    click.echo(f"\n{len(skill_list)} skill(s)")


@skills.command()
@click.argument("skill_dir", type=click.Path(exists=True))
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--scope", default="workspace", type=click.Choice(["user", "workspace", "global"]))
@click.option("--mode", default="server", type=click.Choice(["server", "mirrored", "filesystem"]))
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def push(skill_dir, workspace, scope, mode, server_url, api_key):
    """Parse a skill directory and upload to MemoryLayer."""
    import json
    from pathlib import Path

    import httpx

    from memorylayer.skills import parse_skill_folder

    try:
        manifest, files = parse_skill_folder(Path(skill_dir))
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    payload = {**manifest, "source_mode": mode}
    if workspace:
        payload["workspace_id"] = workspace
    if files:
        payload["files"] = [
            {"path": rel_path, "content": content.decode("utf-8", errors="replace")}
            for rel_path, content in files
        ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{server_url}/v1/skills", content=json.dumps(payload), headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to push skill: {e}", err=True)
        raise SystemExit(1)

    s = data["skill"]
    click.echo(f"Pushed skill '{s['name']}' (id={s['id']}, version={s.get('version', '?')})")


@skills.command()
@click.argument("name")
@click.option("--out", "-o", default=".", show_default=True, help="Output directory")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def pull(name, out, workspace, server_url, api_key):
    """Download a named skill to a local directory."""
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    params = {}
    if workspace:
        params["workspace_id"] = workspace

    try:
        with httpx.Client(timeout=60.0) as client:
            # Resolve by name
            resolve_payload = {"name": name}
            if workspace:
                resolve_payload["workspace_id"] = workspace
            res = client.post(f"{server_url}/v1/skills/resolve", json=resolve_payload, headers=headers)
            res.raise_for_status()
            skill = res.json().get("skill")
            if not skill:
                click.echo(f"Skill '{name}' not found.", err=True)
                raise SystemExit(1)

            skill_id = skill["id"]
            import pathlib
            skill_dir = pathlib.Path(out) / name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Write SKILL.md
            manifest_res = client.get(f"{server_url}/v1/skills/{skill_id}/manifest", headers=headers)
            manifest_res.raise_for_status()
            (skill_dir / "SKILL.md").write_text(manifest_res.text, encoding="utf-8")

            # Write bundle files
            files_res = client.get(f"{server_url}/v1/skills/{skill_id}/files", params=params, headers=headers)
            files_res.raise_for_status()
            for file_info in files_res.json().get("files", []):
                file_path = skill_dir / file_info["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                content_res = client.get(f"{server_url}/v1/skills/{skill_id}/files/{file_info['path']}", headers=headers)
                content_res.raise_for_status()
                file_path.write_bytes(content_res.content)

    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to pull skill: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Pulled skill '{name}' to {skill_dir}")


@skills.command()
@click.option("--target", "-t", default=None, help="Target directory (default: ~/.claude/skills)")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--scope", default=None, help="Filter by scope")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def materialize(target, workspace, scope, server_url, api_key):
    """Bulk pull all visible skills to a local directory for agent harness use."""
    import pathlib

    import httpx

    if target is None:
        target = str(pathlib.Path.home() / ".claude" / "skills")

    target_dir = pathlib.Path(target)
    target_dir.mkdir(parents=True, exist_ok=True)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    params = {}
    if workspace:
        params["workspace_id"] = workspace
    if scope:
        params["scope"] = scope

    try:
        with httpx.Client(timeout=120.0) as client:
            res = client.get(f"{server_url}/v1/skills", params=params, headers=headers)
            res.raise_for_status()
            skill_list = res.json().get("skills", [])

            materialized = 0
            skipped = 0
            for skill in skill_list:
                skill_id = skill["id"]
                skill_name = skill["name"]
                skill_dir = target_dir / skill_name
                hash_file = skill_dir / ".bundle_hash"

                # Skip if already up to date
                server_bundle_hash = skill.get("bundle_hash", "")
                if skill_dir.exists() and hash_file.exists() and server_bundle_hash:
                    existing_hash = hash_file.read_text().strip()
                    if existing_hash == server_bundle_hash:
                        skipped += 1
                        continue

                skill_dir.mkdir(parents=True, exist_ok=True)

                manifest_res = client.get(f"{server_url}/v1/skills/{skill_id}/manifest", headers=headers)
                manifest_res.raise_for_status()
                (skill_dir / "SKILL.md").write_text(manifest_res.text, encoding="utf-8")

                files_res = client.get(f"{server_url}/v1/skills/{skill_id}/files", headers=headers)
                files_res.raise_for_status()
                for file_info in files_res.json().get("files", []):
                    file_path = skill_dir / file_info["path"]
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    content_res = client.get(
                        f"{server_url}/v1/skills/{skill_id}/files/{file_info['path']}", headers=headers
                    )
                    content_res.raise_for_status()
                    file_path.write_bytes(content_res.content)

                if server_bundle_hash:
                    hash_file.write_text(server_bundle_hash)

                materialized += 1

    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to materialize skills: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Materialized {materialized} skill(s) to {target_dir} ({skipped} already up-to-date)")


@skills.command(name="sync")
@click.argument("skill_dir", type=click.Path(exists=True))
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
@click.option("--auto-push", is_flag=True, help="Automatically push local changes on 'push' action")
@click.option("--auto-pull", is_flag=True, help="Automatically pull server changes on 'pull' action")
def skills_sync(skill_dir, workspace, server_url, api_key, auto_push, auto_pull):
    """Reconcile a mirrored skill directory with the server."""
    import hashlib
    import json
    import pathlib

    import httpx

    from memorylayer.skills import parse_skill_folder
    from memorylayer_server.services.skills.frontmatter import render_skill_md

    skill_path = pathlib.Path(skill_dir)

    try:
        manifest, files = parse_skill_folder(skill_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    # Compute local hashes
    frontmatter = {
        "name": manifest["name"],
        "description": manifest.get("description", ""),
        "version": manifest.get("version", "0.1.0"),
    }
    for key in ("license", "compatibility", "allowed_tools"):
        if key in manifest:
            frontmatter[key] = manifest[key]

    local_manifest_text = render_skill_md(frontmatter, manifest.get("body", ""))
    local_manifest_hash = hashlib.sha256(local_manifest_text.encode()).hexdigest()

    # Compute bundle hash: sorted (path, sha256) pairs
    file_hashes = sorted(
        (rel_path, hashlib.sha256(content).hexdigest()) for rel_path, content in files
    )
    bundle_payload = "\n".join(f"{p}:{h}" for p, h in file_hashes)
    local_bundle_hash = hashlib.sha256(bundle_payload.encode()).hexdigest() if file_hashes else ""

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Resolve the skill on the server
    try:
        with httpx.Client(timeout=60.0) as client:
            resolve_payload: dict = {"name": manifest["name"]}
            if workspace:
                resolve_payload["workspace_id"] = workspace
            res = client.post(f"{server_url}/v1/skills/resolve", json=resolve_payload, headers=headers)
            if res.status_code == 404 or not res.json().get("skill"):
                # No server record — local must be pushed
                click.echo(f"Skill '{manifest['name']}' not found on server. Action: push")
                if auto_push:
                    _push_skill(client, server_url, headers, workspace, manifest, files)
                    click.echo("Pushed.")
                return

            res.raise_for_status()
            skill = res.json()["skill"]
            skill_id = skill["id"]

            sync_payload = {
                "manifest_hash": local_manifest_hash,
                "bundle_hash": local_bundle_hash,
            }
            if workspace:
                sync_payload["workspace_id"] = workspace

            sync_res = client.post(
                f"{server_url}/v1/skills/{skill_id}/sync",
                json=sync_payload,
                headers=headers,
            )
            sync_res.raise_for_status()
            result = sync_res.json()

    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    action = result["action"]
    click.echo(f"Skill '{manifest['name']}': {action} — {result['reason']}")

    if action == "push" and auto_push:
        try:
            with httpx.Client(timeout=60.0) as client:
                _push_skill(client, server_url, headers, workspace, manifest, files)
            click.echo("Pushed.")
        except httpx.HTTPError as e:
            click.echo(f"Error during push: {e}", err=True)
            raise SystemExit(1)
    elif action == "pull" and auto_pull:
        import os
        # Re-use pull logic inline
        try:
            with httpx.Client(timeout=60.0) as client:
                manifest_res = client.get(f"{server_url}/v1/skills/{skill_id}/manifest", headers=headers)
                manifest_res.raise_for_status()
                (skill_path / "SKILL.md").write_text(manifest_res.text, encoding="utf-8")

                files_res = client.get(f"{server_url}/v1/skills/{skill_id}/files", headers=headers)
                files_res.raise_for_status()
                for file_info in files_res.json().get("files", []):
                    fp = skill_path / file_info["path"]
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    content_res = client.get(
                        f"{server_url}/v1/skills/{skill_id}/files/{file_info['path']}", headers=headers
                    )
                    content_res.raise_for_status()
                    fp.write_bytes(content_res.content)
            click.echo("Pulled.")
        except httpx.HTTPError as e:
            click.echo(f"Error during pull: {e}", err=True)
            raise SystemExit(1)


def _push_skill(client, server_url, headers, workspace, manifest, files):
    """Helper to push a skill manifest+files via HTTP."""
    import json

    payload = {**manifest, "source_mode": "mirrored"}
    if workspace:
        payload["workspace_id"] = workspace
    if files:
        payload["files"] = [
            {"path": rel_path, "content": content.decode("utf-8", errors="replace")}
            for rel_path, content in files
        ]
    h = {**headers, "Content-Type": "application/json"}
    response = client.post(f"{server_url}/v1/skills", content=json.dumps(payload), headers=h)
    response.raise_for_status()
    return response.json()


@skills.command()
@click.argument("watch_dir", type=click.Path(exists=True))
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def watch(watch_dir, workspace, server_url, api_key):
    """Watch a skill directory and push changes on file save (mirrored mode)."""
    import pathlib

    try:
        from watchfiles import watch as wf_watch
    except ImportError:
        click.echo("Error: 'watchfiles' is required for watch mode. Install it with: pip install watchfiles", err=True)
        raise SystemExit(1)

    import httpx

    from memorylayer.skills import parse_skill_folder

    watch_path = pathlib.Path(watch_dir)
    click.echo(f"Watching {watch_path} for changes (Ctrl+C to stop)...")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        for changes in wf_watch(str(watch_path)):
            click.echo(f"Detected {len(changes)} change(s), pushing...")
            try:
                manifest, files = parse_skill_folder(watch_path)
                import json
                payload = {**manifest, "source_mode": "mirrored"}
                if workspace:
                    payload["workspace_id"] = workspace
                if files:
                    payload["files"] = [
                        {"path": rp, "content": c.decode("utf-8", errors="replace")}
                        for rp, c in files
                    ]
                h = {**headers, "Content-Type": "application/json"}
                with httpx.Client(timeout=30.0) as client:
                    res = client.post(f"{server_url}/v1/skills", content=json.dumps(payload), headers=h)
                    res.raise_for_status()
                click.echo(f"  Pushed '{manifest['name']}' successfully.")
            except Exception as e:
                click.echo(f"  Push failed: {e}", err=True)
    except KeyboardInterrupt:
        click.echo("Watch stopped.")


@skills.command(name="migrate-from-local")
@click.option("--dir", "-d", "scan_dir", default=None, help="Local skills directory (default: ~/.claude/skills)")
@click.option("--scope", "-s", default="user", type=click.Choice(["user", "workspace", "tenant"]), help="Target scope for pushed skills")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--mode", default="server", type=click.Choice(["server", "mirrored"]), help="Storage mode")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def migrate_from_local(scan_dir, scope, workspace, mode, yes, server_url, api_key):
    """Scan a local skills directory and push all skills to MemoryLayer.

    Default scan path: ~/.claude/skills (or MEMORYLAYER_SKILLS_LOCAL_PATHS env var).
    Idempotent: existing skills are updated, new ones are created.
    """
    import json
    import pathlib

    import httpx

    from memorylayer.skills import parse_skill_folder

    env_paths = None
    try:
        import os
        env_paths = os.environ.get("MEMORYLAYER_SKILLS_LOCAL_PATHS")
    except Exception:
        pass

    if scan_dir:
        scan_paths = [pathlib.Path(scan_dir)]
    elif env_paths:
        scan_paths = [pathlib.Path(p.strip()) for p in env_paths.split(":") if p.strip()]
    else:
        scan_paths = [pathlib.Path.home() / ".claude" / "skills"]

    # Gather valid skill directories
    skill_dirs = []
    for base in scan_paths:
        if not base.exists():
            continue
        # Each subdirectory with a SKILL.md is a skill
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "SKILL.md").exists():
                skill_dirs.append(entry)
        # Also check if the base itself is a skill dir
        if (base / "SKILL.md").exists():
            skill_dirs.append(base)

    if not skill_dirs:
        click.echo("No skill directories found.")
        return

    # Parse all skills to show summary
    parsed = []
    for d in skill_dirs:
        try:
            manifest, files = parse_skill_folder(d)
            total_size = sum(len(c) for _, c in files)
            parsed.append((d, manifest, files, total_size))
        except Exception as e:
            click.echo(f"  Warning: Could not parse {d}: {e}", err=True)

    if not parsed:
        click.echo("No valid skills found.")
        return

    click.echo(f"Found {len(parsed)} skill(s) to migrate:")
    for d, manifest, files, total_size in parsed:
        click.echo(f"  {manifest.get('name', d.name)}  ({len(files)} files, {total_size} bytes)")

    if not yes:
        if not click.confirm(f"\nPush {len(parsed)} skill(s) to {server_url} as scope={scope}?"):
            click.echo("Cancelled.")
            return

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ok = 0
    failed = 0
    with httpx.Client(timeout=60.0) as client:
        for d, manifest, files, _ in parsed:
            name = manifest.get("name", d.name)
            try:
                payload = {**manifest, "source_mode": mode}
                if workspace:
                    payload["workspace_id"] = workspace
                if files:
                    payload["files"] = [
                        {"path": rp, "content_b64": __import__("base64").b64encode(c).decode()}
                        for rp, c in files
                    ]
                response = client.post(f"{server_url}/v1/skills", content=json.dumps(payload), headers=headers)
                response.raise_for_status()
                click.echo(f"  OK  {name}")
                ok += 1
            except Exception as e:
                click.echo(f"  FAIL {name}: {e}", err=True)
                failed += 1

    click.echo(f"\nMigration complete: {ok} succeeded, {failed} failed.")


@cli.group()
def mcp():
    """Manage MCP server registry stored in MemoryLayer."""


@mcp.command(name="list")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--transport", default=None, type=click.Choice(["stdio", "http", "sse", "streamable-http"]), help="Filter by transport")
@click.option("--enabled-only/--all", default=True, help="Show only enabled servers (default: enabled only)")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_list(workspace, transport, enabled_only, output_format, server_url, api_key):
    """List registered MCP servers."""
    import httpx

    params: dict = {}
    if workspace:
        params["workspace_id"] = workspace
    if transport:
        params["transport"] = transport
    if enabled_only:
        params["enabled"] = "true"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{server_url}/v1/mcp-servers", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    servers = data.get("mcp_servers", [])
    if output_format == "json":
        click.echo(json.dumps(servers, indent=2))
        return

    if not servers:
        click.echo("No MCP servers found.")
        return

    for s in servers:
        transport_str = s.get("transport", "?")
        location = s.get("command") or s.get("url") or ""
        enabled_marker = "" if s.get("enabled", True) else " [disabled]"
        click.echo(f"  [{transport_str}] {s['name']}{enabled_marker} — {location}  ({s.get('id', '')})")
    click.echo(f"\n{len(servers)} server(s)")


@mcp.command(name="push")
@click.argument("json_file", type=click.Path(exists=True))
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_push(json_file, workspace, server_url, api_key):
    """Import MCP servers from a .mcp.json file into MemoryLayer."""
    import json as _json
    from pathlib import Path

    import httpx

    doc = _json.loads(Path(json_file).read_text(encoding="utf-8"))
    mcp_servers = doc.get("mcpServers", {})

    if not mcp_servers:
        click.echo("No mcpServers entries found in file.")
        return

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers["X-Workspace-ID"] = workspace

    payload = {"mcpServers": mcp_servers}

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{server_url}/v1/mcp-servers/import", content=_json.dumps(payload), headers=headers)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to push MCP servers: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Imported: {result.get('imported', 0)}, Updated: {result.get('updated', 0)}")
    for err in result.get("errors", []):
        click.echo(f"  Error: {err}", err=True)


@mcp.command(name="pull")
@click.option("--output", "-o", default=".mcp.json", show_default=True, help="Output .mcp.json file path")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--reveal-secrets", is_flag=True, help="Include secret values in output (env/headers unmasked)")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_pull(output, workspace, reveal_secrets, server_url, api_key):
    """Export registered MCP servers to a .mcp.json file."""
    import json as _json
    from pathlib import Path

    import httpx

    params: dict = {}
    if workspace:
        params["workspace_id"] = workspace
    if reveal_secrets:
        params["reveal_secrets"] = "true"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers["X-Workspace-ID"] = workspace

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{server_url}/v1/mcp-servers/export", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to pull MCP servers: {e}", err=True)
        raise SystemExit(1)

    out_path = Path(output)
    out_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    count = len(data.get("mcpServers", {}))
    click.echo(f"Pulled {count} server(s) to {out_path}")


@mcp.command(name="sync")
@click.argument("json_file", type=click.Path())
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_sync(json_file, workspace, server_url, api_key):
    """Sync a .mcp.json file with MemoryLayer (push local → server, then pull server → file)."""
    import json as _json
    from pathlib import Path

    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers["X-Workspace-ID"] = workspace

    local_path = Path(json_file)

    # Push local file to server if it exists
    if local_path.exists():
        doc = _json.loads(local_path.read_text(encoding="utf-8"))
        mcp_servers = doc.get("mcpServers", {})
        if mcp_servers:
            payload = {"mcpServers": mcp_servers}
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{server_url}/v1/mcp-servers/import",
                        content=_json.dumps(payload),
                        headers=headers,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                click.echo(f"Pushed: imported={result.get('imported', 0)}, updated={result.get('updated', 0)}")
            except httpx.HTTPError as e:
                click.echo(f"Error during push: {e}", err=True)
                raise SystemExit(1)

    # Pull full server state back to file
    get_headers = {}
    if api_key:
        get_headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        get_headers["X-Workspace-ID"] = workspace

    params: dict = {}
    if workspace:
        params["workspace_id"] = workspace

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{server_url}/v1/mcp-servers/export", params=params, headers=get_headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        click.echo(f"Error during pull: {e}", err=True)
        raise SystemExit(1)

    local_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    count = len(data.get("mcpServers", {}))
    click.echo(f"Synced {count} server(s) → {local_path}")


@mcp.command(name="materialize")
@click.option("--target", "-t", default=None, help="Output .mcp.json file (default: ./.mcp.json)")
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--transport", default=None, type=click.Choice(["stdio", "http", "sse", "streamable-http"]), help="Filter by transport")
@click.option("--write-to-claude-json", "write_claude", is_flag=True, default=False, help="Also write servers to ~/.claude.json (user scope)")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_materialize(target, workspace, transport, write_claude, server_url, api_key):
    """Write all enabled servers to a .mcp.json file (idempotent via manifest_hash)."""
    import json as _json
    from pathlib import Path

    import httpx

    params: dict = {"enabled": "true"}
    if workspace:
        params["workspace_id"] = workspace
    if transport:
        params["transport"] = transport

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{server_url}/v1/mcp-servers/export", params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    out_path = Path(target or ".mcp.json")

    # Idempotency: compare existing file content
    if out_path.exists():
        try:
            existing = _json.loads(out_path.read_text(encoding="utf-8"))
            if existing == data:
                click.echo(f"0 changes — {out_path} already up-to-date.")
                return
        except Exception:
            pass

    out_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    count = len(data.get("mcpServers", {}))
    click.echo(f"Materialized {count} server(s) → {out_path}")

    if write_claude:
        from memorylayer_server.services.mcp_servers.claude_json import write_claude_json_servers
        write_claude_json_servers("user", data.get("mcpServers", {}))
        click.echo("Also wrote to ~/.claude.json (user scope)")


@mcp.command(name="watch")
@click.argument("json_file", type=click.Path())
@click.option("--workspace", "-w", default=None, help="Workspace ID")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_watch(json_file, workspace, server_url, api_key):
    """Watch a .mcp.json file and push changes on save (mirrored mode)."""
    import json as _json
    from pathlib import Path

    try:
        from watchfiles import watch as wf_watch
    except ImportError:
        click.echo("Error: 'watchfiles' is required for watch mode. Install with: pip install watchfiles", err=True)
        raise SystemExit(1)

    import httpx

    watch_path = Path(json_file)
    click.echo(f"Watching {watch_path} for changes (Ctrl+C to stop)...")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers["X-Workspace-ID"] = workspace

    def _push_file():
        if not watch_path.exists():
            click.echo("  File not found, skipping.", err=True)
            return
        doc = _json.loads(watch_path.read_text(encoding="utf-8"))
        mcp_servers = doc.get("mcpServers", {})
        if not mcp_servers:
            click.echo("  No mcpServers entries found, skipping.")
            return
        payload = {"mcpServers": mcp_servers, "source_mode": "mirrored"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{server_url}/v1/mcp-servers/import",
                content=_json.dumps(payload),
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
        click.echo(f"  Pushed: imported={result.get('imported', 0)}, updated={result.get('updated', 0)}")

    try:
        for _changes in wf_watch(str(watch_path.parent)):
            click.echo(f"Detected change, pushing {watch_path.name}...")
            try:
                _push_file()
            except Exception as e:
                click.echo(f"  Push failed: {e}", err=True)
    except KeyboardInterrupt:
        click.echo("Watch stopped.")


@mcp.command(name="migrate-from-local")
@click.option("--workspace", "-w", default=None, help="Target workspace ID")
@click.option("--scope", "-s", default="user", type=click.Choice(["local", "user"]), help="Claude JSON scope to read from")
@click.option("--project-path", default=None, help="Project path key in ~/.claude.json (required for scope=local)")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt")
@click.option("--server-url", default="http://localhost:61001", help="MemoryLayer server URL")
@click.option("--api-key", default=None, help="API key for authentication")
def mcp_migrate_from_local(workspace, scope, project_path, yes, server_url, api_key):
    """Scan ~/.claude.json and bulk-push MCP servers to MemoryLayer.

    Reads from the specified scope (user: top-level mcpServers, local: projects[path].mcpServers).
    Idempotent: existing servers are updated, new ones are created.
    """
    import json as _json

    import httpx

    from memorylayer_server.services.mcp_servers.claude_json import read_claude_json_servers

    servers = read_claude_json_servers(scope, project_path=project_path)

    if not servers:
        click.echo(f"No MCP servers found in ~/.claude.json (scope={scope}).")
        return

    click.echo(f"Found {len(servers)} server(s) to migrate from ~/.claude.json (scope={scope}):")
    for name in servers:
        click.echo(f"  {name}")

    if not yes:
        if not click.confirm(f"\nPush {len(servers)} server(s) to {server_url}?"):
            click.echo("Cancelled.")
            return

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers["X-Workspace-ID"] = workspace

    payload = {"mcpServers": servers}
    if workspace:
        payload["workspace_id"] = workspace

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{server_url}/v1/mcp-servers/import",
                content=_json.dumps(payload),
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
    except httpx.HTTPError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Migration complete: imported={result.get('imported', 0)}, updated={result.get('updated', 0)}")
    for err in result.get("errors", []):
        click.echo(f"  Error: {err}", err=True)


if __name__ == "__main__":
    cli()
