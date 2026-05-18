"""
Knowledgebase Web Viewer -- dev-friendly HTML viewer.

Serves a self-contained HTML page at /kb/{workspace_id} that renders
the latest cached knowledgebase articles with navigation. All rendering
is server-side via the `markdown` library -- no CDN or external deps.

This is a convenience for development/debugging, not a production UI.
"""

import logging
import re

import markdown
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from scitrera_app_framework import Plugin, Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...services._constants import EXT_AUTHORIZATION_SERVICE, EXT_KNOWLEDGEBASE_SERVICE, EXT_STORAGE_BACKEND
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service

router = APIRouter(tags=["knowledgebase-viewer"])

# Shared markdown converter with useful extensions
_md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])


def _md_to_html(text: str) -> str:
    """Convert markdown to HTML, resolving [[wikilinks]] first."""
    # Convert [[target|display]] and [[target]] to HTML links
    def _replace_wikilink(m):
        inner = m.group(1)
        if "|" in inner:
            target, display = inner.split("|", 1)
        else:
            target = display = inner
        # Link to the article by slugified ID
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", target.strip()).strip("-").lower()
        return f'<a href="?article={slug}" class="wikilink">{display.strip()}</a>'

    text = re.sub(r"\[\[([^\]]+)\]\]", _replace_wikilink, text)
    _md.reset()
    return _md.convert(text)


# language=html
_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - MemoryLayer KB</title>
<style>
  :root {{
    --bg: #1a1b26; --surface: #24283b; --border: #3b4261;
    --text: #c0caf5; --text-dim: #565f89; --accent: #7aa2f7;
    --accent-hover: #89b4fa; --heading: #c0caf5;
    --code-bg: #1f2335; --success: #9ece6a; --warn: #e0af68;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; display: flex; min-height: 100vh;
  }}
  nav {{
    width: 280px; min-width: 280px; background: var(--surface);
    border-right: 1px solid var(--border); padding: 1rem;
    overflow-y: auto; position: sticky; top: 0; height: 100vh;
  }}
  nav h2 {{ color: var(--accent); font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.05em; margin: 1rem 0 0.5rem; }}
  nav h2:first-child {{ margin-top: 0; }}
  nav a {{
    display: block; padding: 0.3rem 0.5rem; margin: 0.1rem 0;
    color: var(--text); text-decoration: none; border-radius: 4px;
    font-size: 0.9rem; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }}
  nav a:hover, nav a.active {{ background: var(--border); color: var(--accent); }}
  nav .nav-header {{
    font-size: 1.1rem; font-weight: 700; color: var(--heading);
    padding: 0.5rem; border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
  }}
  nav .nav-meta {{ font-size: 0.75rem; color: var(--text-dim); padding: 0 0.5rem; }}
  main {{
    flex: 1; padding: 2rem 3rem; max-width: 900px; overflow-x: hidden;
  }}
  main h1 {{ color: var(--heading); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; margin-bottom: 1rem; }}
  main h2 {{ color: var(--heading); margin-top: 1.5rem; margin-bottom: 0.5rem; }}
  main h3 {{ color: var(--accent); margin-top: 1.2rem; margin-bottom: 0.4rem; }}
  main p {{ margin-bottom: 0.8rem; }}
  main a {{ color: var(--accent); text-decoration: none; }}
  main a:hover {{ color: var(--accent-hover); text-decoration: underline; }}
  main ul, main ol {{ margin: 0.5rem 0 0.8rem 1.5rem; }}
  main li {{ margin-bottom: 0.2rem; }}
  main code {{
    background: var(--code-bg); padding: 0.15rem 0.4rem; border-radius: 3px;
    font-size: 0.9em;
  }}
  main pre {{ background: var(--code-bg); padding: 1rem; border-radius: 6px;
    overflow-x: auto; margin: 0.8rem 0; }}
  main pre code {{ padding: 0; background: none; }}
  main table {{ border-collapse: collapse; width: 100%; margin: 0.8rem 0; }}
  main th, main td {{ border: 1px solid var(--border); padding: 0.4rem 0.8rem; text-align: left; }}
  main th {{ background: var(--surface); color: var(--heading); }}
  main blockquote {{
    border-left: 3px solid var(--accent); padding: 0.5rem 1rem;
    margin: 0.8rem 0; color: var(--text-dim); background: var(--surface);
    border-radius: 0 4px 4px 0;
  }}
  .empty-state {{
    text-align: center; padding: 4rem 2rem; color: var(--text-dim);
  }}
  .empty-state h2 {{ color: var(--heading); margin-bottom: 1rem; }}
  .empty-state code {{ color: var(--accent); }}
  .badge {{
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.3rem;
  }}
  .badge-community {{ background: #1e3a5f; color: var(--accent); }}
  .badge-entity {{ background: #2d3b1f; color: var(--success); }}
  .badge-index {{ background: #3b2f1f; color: var(--warn); }}
</style>
</head>
<body>
<nav>
  <div class="nav-header">MemoryLayer KB</div>
  <div class="nav-meta">{workspace_id}</div>
  {nav_html}
</nav>
<main>
  {content_html}
</main>
</body>
</html>
"""


def _build_nav(articles: list[dict], active_id: str | None) -> str:
    """Build the sidebar navigation HTML from article list."""
    index_links = []
    community_links = []
    entity_links = []

    for a in articles:
        aid = a.get("article_id", "")
        title = a.get("title", aid)
        atype = a.get("article_type", "")
        active_cls = ' class="active"' if aid == active_id else ""

        badge = ""
        if atype == "community":
            badge = ' <span class="badge badge-community">C</span>'
        elif atype == "entity":
            badge = ' <span class="badge badge-entity">E</span>'
        elif atype == "index":
            badge = ' <span class="badge badge-index">I</span>'

        link = f'<a href="?article={aid}"{active_cls}>{title}{badge}</a>'

        if atype == "index":
            index_links.append(link)
        elif atype == "community":
            community_links.append(link)
        else:
            entity_links.append(link)

    parts = []
    if index_links:
        parts.append("<h2>Overview</h2>" + "\n".join(index_links))
    if community_links:
        parts.append(f"<h2>Communities ({len(community_links)})</h2>" + "\n".join(community_links))
    if entity_links:
        parts.append(f"<h2>Entities ({len(entity_links)})</h2>" + "\n".join(entity_links))

    return "\n".join(parts)


# language=html
_WORKSPACE_LIST_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemoryLayer Knowledgebases</title>
<style>
  :root {{
    --bg: #1a1b26; --surface: #24283b; --border: #3b4261;
    --text: #c0caf5; --text-dim: #565f89; --accent: #7aa2f7;
    --accent-hover: #89b4fa; --heading: #c0caf5;
    --success: #9ece6a; --warn: #e0af68;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
    display: flex; justify-content: center; padding: 2rem;
  }}
  .container {{ max-width: 700px; width: 100%; }}
  h1 {{ color: var(--heading); margin-bottom: 0.5rem; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 2rem; font-size: 0.95rem; }}
  .workspace-list {{ list-style: none; }}
  .workspace-item {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 0.75rem; overflow: hidden;
    transition: border-color 0.15s;
  }}
  .workspace-item:hover {{ border-color: var(--accent); }}
  .workspace-item a {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.25rem; color: var(--text); text-decoration: none;
  }}
  .workspace-item a:hover {{ color: var(--accent); }}
  .ws-name {{ font-weight: 600; font-size: 1.05rem; }}
  .ws-meta {{ font-size: 0.8rem; color: var(--text-dim); }}
  .ws-badge {{
    padding: 0.2rem 0.6rem; border-radius: 10px; font-size: 0.75rem;
    font-weight: 600; white-space: nowrap;
  }}
  .ws-badge-kb {{ background: #1e3a5f; color: var(--accent); }}
  .ws-badge-none {{ background: var(--border); color: var(--text-dim); }}
  .empty-state {{
    text-align: center; padding: 3rem; color: var(--text-dim);
    background: var(--surface); border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .empty-state h2 {{ color: var(--heading); margin-bottom: 0.5rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>MemoryLayer Knowledgebases</h1>
  <p class="subtitle">Select a workspace to view its knowledgebase</p>
  {content}
</div>
</body>
</html>
"""


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@router.get("/kb", response_class=HTMLResponse)
async def kb_workspace_list(
    request: Request,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: logging.Logger = Depends(get_logger),
) -> HTMLResponse:
    """
    List all accessible workspaces with links to their knowledgebases.

    Auth-aware: uses get_allowed_workspaces() to filter the list,
    so enterprise deployments only show workspaces the user can access.
    """
    storage = get_extension(EXT_STORAGE_BACKEND, v)

    # Build auth context (no body needed)
    try:
        ctx = await auth_service.build_context(request, None)
    except Exception:
        # If auth fails, proceed with defaults for the workspace list
        ctx = None

    # Get all workspaces from storage
    try:
        workspaces = await storage.list_workspaces()
    except Exception as e:
        logger.error("Failed to list workspaces for KB viewer: %s", e)
        workspaces = []

    # Filter by authorization if auth context is available
    if ctx and authz_service:
        try:
            allowed = await authz_service.get_allowed_workspaces(
                tenant_id=getattr(ctx, "tenant_id", "_default"),
                user_id=getattr(ctx, "user_id", "anonymous"),
            )
            # ['*'] means all workspaces are accessible
            if allowed != ["*"]:
                allowed_set = set(allowed)
                workspaces = [w for w in workspaces if w.id in allowed_set]
        except Exception as e:
            logger.debug("Authorization check for workspace list failed (allowing all): %s", e)

    if not workspaces:
        return HTMLResponse(
            _WORKSPACE_LIST_TEMPLATE.format(
                content=(
                    '<div class="empty-state">'
                    "<h2>No workspaces found</h2>"
                    "<p>Create a workspace and generate a knowledgebase first.</p>"
                    "</div>"
                ),
            )
        )

    # Check which workspaces have KB articles
    ws_kb_status: dict[str, int] = {}
    for w in workspaces:
        try:
            articles = await storage.list_kb_articles(workspace_id=w.id, limit=1)
            ws_kb_status[w.id] = len(articles)
        except (NotImplementedError, Exception):
            ws_kb_status[w.id] = 0

    # Build workspace list HTML
    items = []
    for w in workspaces:
        has_kb = ws_kb_status.get(w.id, 0) > 0
        badge_cls = "ws-badge-kb" if has_kb else "ws-badge-none"
        badge_text = "KB Available" if has_kb else "No KB"
        name = getattr(w, "name", None) or w.id
        items.append(
            f'<li class="workspace-item">'
            f'<a href="/kb/{w.id}">'
            f'<div><div class="ws-name">{name}</div>'
            f'<div class="ws-meta">{w.id}</div></div>'
            f'<span class="ws-badge {badge_cls}">{badge_text}</span>'
            f"</a></li>"
        )

    content = f'<ul class="workspace-list">{"".join(items)}</ul>'
    return HTMLResponse(_WORKSPACE_LIST_TEMPLATE.format(content=content))


@router.get("/kb/{workspace_id}", response_class=HTMLResponse)
async def kb_viewer(
    request: Request,
    workspace_id: str,
    article: str | None = None,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> HTMLResponse:
    """
    Self-contained HTML viewer for a workspace's knowledgebase.

    Renders the latest cached articles. Pass ``?article=<id>`` to view
    a specific article; defaults to the index.
    """
    storage = get_extension(EXT_STORAGE_BACKEND, v)

    # Load all articles for this workspace
    try:
        articles = await storage.list_kb_articles(workspace_id=workspace_id, limit=500)
    except NotImplementedError:
        return HTMLResponse(
            _PAGE_TEMPLATE.format(
                title="Not Available",
                workspace_id=workspace_id,
                nav_html="",
                content_html='<div class="empty-state"><h2>Storage backend does not support knowledgebase</h2></div>',
            ),
            status_code=501,
        )

    if not articles:
        return HTMLResponse(
            _PAGE_TEMPLATE.format(
                title="No Knowledgebase",
                workspace_id=workspace_id,
                nav_html="",
                content_html=(
                    '<div class="empty-state">'
                    "<h2>No knowledgebase generated yet</h2>"
                    "<p>Generate one via the API:</p>"
                    '<p><code>POST /v1/knowledgebase/generate</code></p>'
                    "</div>"
                ),
            ),
        )

    # Find the requested article (default to index)
    active_id = article or "index"
    current = None
    for a in articles:
        if a.get("article_id") == active_id:
            current = a
            break

    # Fall back to first article if requested one not found
    if current is None:
        current = articles[0]
        active_id = current.get("article_id", "")

    # Render
    title = current.get("title", active_id)
    content_md = current.get("content_md", "*No content*")
    content_html = _md_to_html(content_md)
    nav_html = _build_nav(articles, active_id)

    return HTMLResponse(
        _PAGE_TEMPLATE.format(
            title=title,
            workspace_id=workspace_id,
            nav_html=nav_html,
            content_html=content_html,
        )
    )


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class KBViewerPlugin(Plugin):
    """Plugin to register the KB web viewer route."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # multi-extension

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
