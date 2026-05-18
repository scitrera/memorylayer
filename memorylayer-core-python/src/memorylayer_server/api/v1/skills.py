"""
Skills API endpoints.

Endpoints:
- POST   /v1/skills              - Create a skill
- GET    /v1/skills              - List skills
- GET    /v1/skills/{id}         - Get a skill (manifest fields)
- GET    /v1/skills/{id}/manifest - Render full SKILL.md text
- GET    /v1/skills/{id}/files   - List bundle files
- GET    /v1/skills/{id}/files/{path} - Stream a single file
- PUT    /v1/skills/{id}         - Update manifest fields
- PUT    /v1/skills/{id}/files/{path} - Upsert one file
- DELETE /v1/skills/{id}         - Delete skill (cascades to files)
- POST   /v1/skills/resolve      - Resolve skill by name (precedence) or query (vector search)
- POST   /v1/skills/{id}/sync    - Reconcile mirrored skill via hash comparison
- GET    /v1/skills/{id}/bundle  - Stream skill bundle as NDJSON or tar.gz
"""

import base64
import io
import json
import logging
import tarfile
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.lifecycle.fastapi import get_logger

from ...models.memory import MemoryType, RecallInput
from ...models.skill import Skill, SkillCreateInput, SkillUpdateInput
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.memory import MemoryService
from ...services.skills import SkillsService
from ...services.skills.frontmatter import render_skill_md
from ...services.skills.resolution import RequestContext, SkillsResolutionService
from ...services.skills.sync import compute_sync_action
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service, get_memory_service, get_skills_resolution_service, get_skills_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/skills", tags=["skills"])

# ── Request / Response schemas ────────────────────────────────────────────────


class SkillFileInfo(BaseModel):
    path: str
    kind: str
    size_bytes: int
    content_hash: str
    mime_type: str | None = None


class SkillResponse(BaseModel):
    skill: Skill


class SkillListResponse(BaseModel):
    skills: list[Skill]
    total_count: int


class SkillFilesListResponse(BaseModel):
    files: list[SkillFileInfo]


class SkillFileUpsertRequest(BaseModel):
    content_b64: str | None = Field(None, description="Base64-encoded file content")
    mime_type: str | None = None


class SkillResolveRequest(BaseModel):
    name: str | None = Field(None, description="Exact skill name — returns precedence winner")
    query: str | None = Field(None, description="Intent query — runs vector recall against skill memories")
    scope_hint: str | None = Field(None, description="Restrict resolution to a single scope: 'user', 'workspace', or 'global'")
    workspace_id: str | None = Field(
        None, description="Workspace to resolve against; defaults to the authenticated context's workspace."
    )


class SkillResolveResponse(BaseModel):
    skill: Skill | None = None
    candidates: list[Skill] = Field(default_factory=list, description="Populated for query-based resolution")


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=SkillResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_skill(
    http_request: Request,
    request: SkillCreateInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillResponse:
    """Create a new skill."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "write", workspace_id=workspace_id)

        skill = await skills_service.create_skill(
            input=request,
            workspace_id=workspace_id,
            tenant_id=getattr(ctx, "tenant_id", ""),
            user_id=request.user_id or getattr(ctx, "user_id", None),
        )
        return SkillResponse(skill=skill)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create skill: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create skill")


@router.get(
    "",
    response_model=SkillListResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def list_skills(
    http_request: Request,
    workspace_id: str | None = Query(None),
    name: str | None = Query(None),
    enabled: bool | None = Query(None),
    include_shadowed: bool = Query(False, description="Return all skills including shadowed duplicates"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    resolution_service: SkillsResolutionService = Depends(get_skills_resolution_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillListResponse:
    """List skills for a workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skills = await skills_service.list_skills(
            workspace_id=workspace_id,
            name=name,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )

        if not include_shadowed:
            resolution_ctx = RequestContext(
                workspace_id=workspace_id,
                user_id=getattr(ctx, "user_id", None),
                tenant_id=getattr(ctx, "tenant_id", ""),
            )
            skills = resolution_service.apply_shadowing(skills, resolution_ctx)

        return SkillListResponse(skills=skills, total_count=len(skills))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list skills: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list skills")


@router.post(
    "/resolve",
    response_model=SkillResolveResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def resolve_skill(
    http_request: Request,
    request: SkillResolveRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    resolution_service: SkillsResolutionService = Depends(get_skills_resolution_service),
    memory_service: MemoryService = Depends(get_memory_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillResolveResponse:
    """Resolve a skill by name (precedence) or query (vector intent search)."""
    if not request.name and not request.query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'name' or 'query' must be provided")
    try:
        # Pass ``request`` (not None) so build_context reads workspace_id from
        # the body — otherwise resolve_workspace falls through to
        # ``DEFAULT_WORKSPACE_ID="_default"`` and the OBO grant scope check
        # raises 403 even when the caller correctly populated the body.
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        resolution_ctx = RequestContext(
            workspace_id=workspace_id,
            user_id=getattr(ctx, "user_id", None),
            tenant_id=getattr(ctx, "tenant_id", ""),
        )

        if request.name:
            skill = await resolution_service.resolve(request.name, resolution_ctx, scope_hint=request.scope_hint)
            return SkillResolveResponse(skill=skill)

        # query-based: recall procedural memories with subtype=skill, look up skill records
        recall_result = await memory_service.recall(
            workspace_id=workspace_id,
            input=RecallInput(
                query=request.query,
                types=[MemoryType.PROCEDURAL],
                subtypes=["skill"],
                limit=10,
            ),
        )
        candidates = []
        seen_ids: set[str] = set()
        for mem in recall_result.memories:
            skill_id = mem.metadata.get("skill_id")
            if skill_id and skill_id not in seen_ids:
                seen_ids.add(skill_id)
                skill = await skills_service.get_skill(workspace_id, skill_id)
                if skill:
                    candidates.append(skill)
        return SkillResolveResponse(skill=candidates[0] if candidates else None, candidates=candidates)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve skill: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to resolve skill")


@router.get(
    "/{skill_id}",
    response_model=SkillResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_skill(
    http_request: Request,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillResponse:
    """Get a skill by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")
        return SkillResponse(skill=skill)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get skill")


@router.get(
    "/{skill_id}/manifest",
    responses={
        200: {"content": {"text/markdown": {}}},
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_skill_manifest(
    http_request: Request,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> Response:
    """Render the full SKILL.md text for a skill."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        frontmatter: dict[str, Any] = {"name": skill.name, "description": skill.description, "version": skill.version}
        if skill.license:
            frontmatter["license"] = skill.license
        if skill.compatibility:
            frontmatter["compatibility"] = skill.compatibility
        if skill.allowed_tools:
            frontmatter["allowed-tools"] = skill.allowed_tools
        if skill.metadata:
            frontmatter["metadata"] = str(skill.metadata)

        text = render_skill_md(frontmatter, skill.body)
        return Response(content=text, media_type="text/markdown")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to render manifest for skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to render manifest")


@router.get(
    "/{skill_id}/files",
    response_model=SkillFilesListResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def list_skill_files(
    http_request: Request,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillFilesListResponse:
    """List files in a skill bundle."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        files = await skills_service.list_files(skill_id)
        file_infos = [
            SkillFileInfo(
                path=f.path,
                kind=f.kind,
                size_bytes=f.size_bytes,
                content_hash=f.content_hash,
                mime_type=f.mime_type,
            )
            for f in files
        ]
        return SkillFilesListResponse(files=file_infos)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list files for skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list skill files")


@router.get(
    "/{skill_id}/bundle",
    responses={
        200: {"content": {"application/x-ndjson": {}, "application/gzip": {}}},
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_skill_bundle(
    http_request: Request,
    skill_id: str,
    format: str = Query("ndjson", description="Bundle format: 'ndjson' or 'tar.gz'"),
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    """Stream a skill's file bundle as NDJSON (default) or tar.gz."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        files = await skills_service.list_files(skill_id)

        if format == "tar.gz":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for sf in files:
                    content_bytes = sf.content if isinstance(sf.content, bytes) else sf.content.encode()
                    info = tarfile.TarInfo(name=sf.path)
                    info.size = len(content_bytes)
                    tar.addfile(info, io.BytesIO(content_bytes))
            buf.seek(0)
            return StreamingResponse(
                iter([buf.read()]),
                media_type="application/gzip",
                headers={"Content-Disposition": f'attachment; filename="{skill.name}.tar.gz"'},
            )

        async def _ndjson_stream():
            header = {"type": "header", "skill_id": skill_id, "skill_name": skill.name, "version": skill.version, "file_count": len(files)}
            yield json.dumps(header) + "\n"
            for sf in files:
                content_bytes = sf.content if isinstance(sf.content, bytes) else sf.content.encode()
                line = {
                    "type": "file",
                    "path": sf.path,
                    "kind": sf.kind,
                    "content_b64": base64.b64encode(content_bytes).decode(),
                    "content_hash": sf.content_hash,
                    "size_bytes": sf.size_bytes,
                    "mime_type": sf.mime_type,
                }
                yield json.dumps(line) + "\n"
            footer = {"type": "footer", "file_count": len(files), "bundle_hash": skill.bundle_hash}
            yield json.dumps(footer) + "\n"

        return StreamingResponse(
            _ndjson_stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{skill.name}.ndjson"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stream bundle for skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to stream bundle")


@router.get(
    "/{skill_id}/files/{file_path:path}",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_skill_file(
    http_request: Request,
    skill_id: str,
    file_path: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    """Stream a single file from a skill bundle."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        sf = await skills_service.get_file(skill_id, file_path)
        if not sf:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {file_path} not found")

        media_type = sf.mime_type or "application/octet-stream"
        return StreamingResponse(iter([sf.content]), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get file %s from skill %s: %s", file_path, skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get skill file")


@router.put(
    "/{skill_id}",
    response_model=SkillResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def update_skill(
    http_request: Request,
    skill_id: str,
    request: SkillUpdateInput,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillResponse:
    """Update manifest fields of a skill."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "write", workspace_id=workspace_id)

        skill = await skills_service.update_skill(workspace_id, skill_id, request)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")
        return SkillResponse(skill=skill)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to update skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update skill")


@router.put(
    "/{skill_id}/files/{file_path:path}",
    response_model=SkillFileInfo,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def upsert_skill_file(
    http_request: Request,
    skill_id: str,
    file_path: str,
    request: SkillFileUpsertRequest,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillFileInfo:
    """Upsert a file in a skill bundle."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "write", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        if not request.content_b64:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="content_b64 is required")

        content = base64.b64decode(request.content_b64)
        sf = await skills_service.upsert_file(
            skill_id=skill_id,
            path=file_path,
            content=content,
            mime_type=request.mime_type,
            workspace_id=workspace_id,
        )
        return SkillFileInfo(
            path=sf.path,
            kind=sf.kind,
            size_bytes=sf.size_bytes,
            content_hash=sf.content_hash,
            mime_type=sf.mime_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upsert file %s for skill %s: %s", file_path, skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upsert skill file")


@router.delete(
    "/{skill_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def delete_skill(
    http_request: Request,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """Delete a skill and all its files."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "write", workspace_id=workspace_id)

        deleted = await skills_service.delete_skill(workspace_id, skill_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete skill")


class SkillSyncRequest(BaseModel):
    manifest_hash: str = Field("", description="SHA-256 of local SKILL.md")
    bundle_hash: str = Field("", description="SHA-256 of local bundle files")
    workspace_id: str | None = None


class SkillSyncResponse(BaseModel):
    action: str = Field(description="push | pull | conflict | in_sync")
    reason: str
    server_manifest_hash: str
    server_bundle_hash: str


@router.post(
    "/{skill_id}/sync",
    response_model=SkillSyncResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def sync_skill(
    http_request: Request,
    skill_id: str,
    request: SkillSyncRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> SkillSyncResponse:
    """Reconcile a mirrored skill: compare client hashes with server state."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "read", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        result = compute_sync_action(
            server_manifest_hash=skill.manifest_hash,
            server_bundle_hash=skill.bundle_hash,
            client_manifest_hash=request.manifest_hash,
            client_bundle_hash=request.bundle_hash,
        )
        return SkillSyncResponse(
            action=result.action,
            reason=result.reason,
            server_manifest_hash=result.server_manifest_hash,
            server_bundle_hash=result.server_bundle_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to sync skill %s: %s", skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to sync skill")


class SkillsAPIPlugin(Plugin):
    """Plugin to register skills API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
