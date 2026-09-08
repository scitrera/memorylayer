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
- DELETE /v1/skills/{id}/files/{path} - Delete one file (idempotent)
- PUT    /v1/skills/{id}/manifest - Conditionally replace the full manifest
- GET    /v1/skills/{id}/revisions - List immutable manifest revisions
- POST   /v1/skills/{id}/delete  - Conditionally tombstone a skill
- POST   /v1/skills/{id}/restore - Conditionally restore a skill
- DELETE /v1/skills/{id}         - Tombstone a skill (retains files for restore)
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
from ...models.skill import Skill, SkillCreateInput, SkillReplaceInput, SkillRevision, SkillUpdateInput
from ...models.versioned_resource import VersionedResourcePreconditionFailedError
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.memory import MemoryService
from ...services.skills import SkillsService
from ...services.skills.addenda import (
    SKILL_ADDENDUM_ACCEPTED_STATUS,
    SKILL_ADDENDUM_SUBTYPE,
    compose_skill_with_addenda,
    compose_skills_with_addenda,
)
from ...services.skills.frontmatter import render_skill_md
from ...services.skills.resolution import RequestContext, SkillsResolutionService
from ...services.skills.sync import compute_sync_action
from .. import EXT_MULTI_API_ROUTERS
from ._versioned_resource import raise_api_error as _shared_raise_api_error
from ._versioned_resource import required_header as _required_header
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
    replayed: bool = False


class SkillRevisionResponse(BaseModel):
    skill: Skill
    action: str
    operation_id: str


class SkillRevisionListResponse(BaseModel):
    revisions: list[SkillRevisionResponse]
    next_page_token: str | None = None


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
    workspace_id: str | None = Field(None, description="Workspace to resolve against; defaults to the authenticated context's workspace.")
    include_addenda: bool = Field(False, description="Attach accepted skill_addendum memories to returned skill bodies")


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
    response: Response,
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

        # `user_id` is the user-PRIVATE owner (scope), NOT authorship. It must stay
        # NULL for shared workspace/global skills — a non-null user_id makes the skill
        # user-private and drops it from the `_global` union (which filters
        # `user_id IS NULL`), so it vanishes from every workspace listing. Take it
        # ONLY from the explicit request field; never fall back to the caller's
        # identity (that conflation is what hid tenant-shared seeded skills).
        # Authorship is recorded separately as provenance in metadata.
        actor = getattr(ctx, "user_id", None)
        if actor:
            request.metadata = {**(request.metadata or {}), "created_by": actor, "updated_by": actor}

        operation_id = http_request.headers.get("Idempotency-Key", "").strip()
        expected = http_request.headers.get("If-None-Match", "").strip()
        if operation_id or expected:
            operation_id = _required_header(http_request, "Idempotency-Key")
            expected = _required_header(http_request, "If-None-Match")
            if expected != "*":
                raise VersionedResourcePreconditionFailedError(
                    "create requires If-None-Match: *"
                )
            if request.files:
                raise ValueError(
                    "conditional manifest create does not accept bundle files; "
                    "upload child files after the manifest is committed"
                )
            result = await skills_service.create_skill_versioned(
                input=request,
                workspace_id=workspace_id,
                tenant_id=getattr(ctx, "tenant_id", ""),
                user_id=request.user_id,
                operation_id=operation_id,
                expected_etag=expected,
            )
            response.headers["ETag"] = result.skill.etag
            return SkillResponse(skill=result.skill, replayed=result.replayed)

        skill = await skills_service.create_skill(
            input=request,
            workspace_id=workspace_id,
            tenant_id=getattr(ctx, "tenant_id", ""),
            user_id=request.user_id,
        )
        response.headers["ETag"] = skill.etag
        return SkillResponse(skill=skill)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        _raise_api_error(e, "create skill")


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
    include_addenda: bool = Query(False, description="Attach accepted skill_addendum memories to returned skill bodies"),
    include_global: bool = Query(True, description="Union tenant-shared _global skills into the workspace listing"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    resolution_service: SkillsResolutionService = Depends(get_skills_resolution_service),
    memory_service: MemoryService = Depends(get_memory_service),
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
            include_global=include_global,
        )

        if not include_shadowed:
            resolution_ctx = RequestContext(
                workspace_id=workspace_id,
                user_id=getattr(ctx, "user_id", None),
                tenant_id=getattr(ctx, "tenant_id", ""),
            )
            skills = resolution_service.apply_shadowing(skills, resolution_ctx)

        if include_addenda:
            skills = await compose_skills_with_addenda(memory_service.storage, skills)

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
            if skill and request.include_addenda:
                skill = await compose_skill_with_addenda(memory_service.storage, skill)
            return SkillResolveResponse(skill=skill)

        # query-based: recall procedural memories with subtype=skill, look up skill records
        recall_result = await memory_service.recall(
            workspace_id=workspace_id,
            input=RecallInput(
                query=request.query,
                types=[MemoryType.PROCEDURAL],
                subtypes=["skill", SKILL_ADDENDUM_SUBTYPE],
                limit=10,
            ),
        )
        candidates = []
        seen_ids: set[str] = set()
        for mem in recall_result.memories:
            if mem.subtype == SKILL_ADDENDUM_SUBTYPE and mem.metadata.get("status") != SKILL_ADDENDUM_ACCEPTED_STATUS:
                continue
            skill_id = mem.metadata.get("skill_id")
            if skill_id and skill_id not in seen_ids:
                seen_ids.add(skill_id)
                skill = await skills_service.get_skill(workspace_id, skill_id)
                if skill:
                    candidates.append(skill)
        if request.include_addenda:
            candidates = await compose_skills_with_addenda(memory_service.storage, candidates)
        return SkillResolveResponse(skill=candidates[0] if candidates else None, candidates=candidates)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve skill: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to resolve skill")


@router.get(
    "/{skill_id}/revisions",
    response_model=SkillRevisionListResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def list_skill_revisions(
    http_request: Request,
    skill_id: str,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
) -> SkillRevisionListResponse:
    """List immutable native skill-manifest revisions newest first."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "skills", "read", workspace_id=workspace_id
        )
        revisions, next_token = await skills_service.list_revision_page(
            ctx.tenant_id,
            workspace_id,
            skill_id,
            limit=limit,
            page_token=page_token,
        )
        if not revisions and await skills_service.get_skill(
            workspace_id, skill_id, include_deleted=True
        ) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Skill {skill_id} not found",
            )
        return SkillRevisionListResponse(
            revisions=[_to_revision(item) for item in revisions],
            next_page_token=next_token,
        )
    except Exception as exc:
        _raise_api_error(exc, "list skill revisions")


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
    response: Response,
    skill_id: str,
    workspace_id: str | None = Query(None),
    include_deleted: bool = Query(False),
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

        skill = await skills_service.get_skill(
            workspace_id,
            skill_id,
            include_deleted=include_deleted,
        )
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")
        response.headers["ETag"] = skill.etag
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
    include_addenda: bool = Query(False, description="Attach accepted skill_addendum memories to the rendered manifest"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    memory_service: MemoryService = Depends(get_memory_service),
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
        if include_addenda:
            skill = await compose_skill_with_addenda(memory_service.storage, skill)

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


@router.put(
    "/{skill_id}/manifest",
    response_model=SkillResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        412: {"model": ErrorResponse},
        428: {"model": ErrorResponse},
    },
)
async def replace_skill_manifest(
    http_request: Request,
    response: Response,
    skill_id: str,
    request: SkillReplaceInput,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
) -> SkillResponse:
    """Conditionally replace a skill's complete semantic manifest."""
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "skills", "write", workspace_id=workspace_id
        )
        result = await skills_service.replace_skill_versioned(
            workspace_id,
            skill_id,
            request,
            tenant_id=ctx.tenant_id,
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.skill.etag
        return SkillResponse(skill=result.skill, replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "replace skill manifest")


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
) -> Response:
    """Return a single file from a skill bundle.

    Uses a BUFFERED Response (Content-Length) rather than StreamingResponse: the
    file content is already fully in memory, and chunked/streamed responses do not
    survive the Aether ProxyHttp relay (the body arrives empty), whereas a
    Content-Length response does. Buffering is both correct here and relay-safe."""
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
        content = sf.content if isinstance(sf.content, bytes) else sf.content.encode()
        return Response(content=content, media_type=media_type)

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
    "/{skill_id}/files/{file_path:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def delete_skill_file(
    http_request: Request,
    skill_id: str,
    file_path: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """Delete a single file from a skill bundle.

    Mirrors the PUT upsert route's auth/ACL/workspace scoping. The skill-file
    store removes the file and the skill's ``bundle_hash`` is recomputed so the
    record stays consistent (handled by ``SkillsService.delete_file``). Deleting
    a path that is already absent is idempotent (returns 204) — this lets the
    SDK's ``save()`` reconcile loop prune dropped manifest files without racing.
    A missing *skill* still 404s, matching the other skill-file routes.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "skills", "write", workspace_id=workspace_id)

        skill = await skills_service.get_skill(workspace_id, skill_id)
        if not skill:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill {skill_id} not found")

        # delete_file recomputes bundle_hash and cleans reference memories;
        # absent files return False, which we treat as idempotent success.
        await skills_service.delete_file(skill_id=skill_id, path=file_path, workspace_id=workspace_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete file %s for skill %s: %s", file_path, skill_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete skill file")


@router.post(
    "/{skill_id}/delete",
    response_model=SkillResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        412: {"model": ErrorResponse},
        428: {"model": ErrorResponse},
    },
)
async def delete_skill_manifest(
    http_request: Request,
    response: Response,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
) -> SkillResponse:
    """Conditionally write a durable skill-manifest tombstone."""
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "skills", "write", workspace_id=workspace_id
        )
        result = await skills_service.delete_skill_versioned(
            workspace_id,
            skill_id,
            tenant_id=ctx.tenant_id,
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.skill.etag
        return SkillResponse(skill=result.skill, replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "delete skill manifest")


@router.post(
    "/{skill_id}/restore",
    response_model=SkillResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        412: {"model": ErrorResponse},
        428: {"model": ErrorResponse},
    },
)
async def restore_skill_manifest(
    http_request: Request,
    response: Response,
    skill_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    skills_service: SkillsService = Depends(get_skills_service),
) -> SkillResponse:
    """Conditionally reactivate a native skill tombstone."""
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "skills", "write", workspace_id=workspace_id
        )
        result = await skills_service.restore_skill_versioned(
            workspace_id,
            skill_id,
            tenant_id=ctx.tenant_id,
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.skill.etag
        return SkillResponse(skill=result.skill, replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "restore skill manifest")


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
    """Legacy unconditional delete, implemented as a durable tombstone."""
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


def _to_revision(revision: SkillRevision) -> SkillRevisionResponse:
    return SkillRevisionResponse(
        skill=revision.skill,
        action=revision.action,
        operation_id=revision.operation_id,
    )


def _raise_api_error(exc: Exception, operation: str) -> None:
    _shared_raise_api_error(exc, operation, __name__)


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
