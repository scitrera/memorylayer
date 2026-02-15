"""
Workspace management API endpoints.

Endpoints:
- POST /v1/workspaces - Create workspace
- GET /v1/workspaces/{workspace_id} - Get workspace
- PUT /v1/workspaces/{workspace_id} - Update workspace
"""
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import StreamingResponse
from scitrera_app_framework import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger

from .schemas import (
    WorkspaceCreateRequest,
    WorkspaceUpdateRequest,
    WorkspaceResponse,
    WorkspaceListResponse,
    ErrorResponse,
    MemoryExportItem,
    AssociationExportItem,
    WorkspaceExportData,
    WorkspaceImportRequest,
    WorkspaceImportResult,
)
from ...services.workspace import WorkspaceService
from ...services.ontology import get_ontology_service as _get_ontology_service
from ...services.memory import MemoryService
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from .deps import get_auth_service, get_authz_service, get_workspace_service, get_memory_service

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_workspace(
        http_request: Request,
        request: WorkspaceCreateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Create a new workspace.

    Workspaces provide tenant-level memory isolation.

    Args:
        http_request: FastAPI request (for headers)
        request: Workspace creation request
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Created workspace

    Raises:
        HTTPException: If workspace creation fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "workspaces", "create")

        # Generate workspace ID
        workspace_id = f"ws_{uuid4().hex[:16]}"

        logger.info(
            "Creating workspace: %s for tenant: %s, name: %s",
            workspace_id,
            ctx.tenant_id,
            request.name
        )

        # Create workspace
        from ...models.workspace import Workspace
        workspace = Workspace(
            id=workspace_id,
            tenant_id=ctx.tenant_id,
            name=request.name,
            settings=request.settings,
        )

        # Store workspace via workspace service
        workspace = await workspace_service.create_workspace(workspace)

        logger.info("Created workspace: %s", workspace_id)
        return WorkspaceResponse(workspace=workspace)

    except ValueError as e:
        logger.warning("Invalid workspace creation request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create workspace: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workspace"
        )


@router.get(
    "",
    response_model=WorkspaceListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_workspaces(
        http_request: Request,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceListResponse:
    """
    List all workspaces.

    Returns:
        List of all workspaces
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "workspaces", "read")

        logger.debug("Listing workspaces")
        workspaces = await workspace_service.list_workspaces()

        return WorkspaceListResponse(workspaces=workspaces)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list workspaces: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workspaces"
        )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_workspace(
        http_request: Request,
        workspace_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Retrieve a workspace by ID.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Workspace object

    Raises:
        HTTPException: If workspace not found or access denied
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "read",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.debug("Getting workspace: %s", workspace_id)

        # Get workspace via workspace service
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        return WorkspaceResponse(workspace=workspace)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace"
        )


@router.put(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def update_workspace(
        http_request: Request,
        workspace_id: str,
        request: WorkspaceUpdateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Update an existing workspace.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        request: Workspace update request
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Updated workspace

    Raises:
        HTTPException: If workspace not found or update fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "write",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.info("Updating workspace: %s", workspace_id)

        # Get existing workspace
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        # Update fields
        if request.name is not None:
            workspace = workspace.model_copy(update={"name": request.name})
        if request.settings is not None:
            workspace = workspace.model_copy(update={"settings": request.settings})
        workspace = await workspace_service.update_workspace(workspace)
        return WorkspaceResponse(workspace=workspace)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid workspace update request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workspace"
        )


@router.get(
    "/{workspace_id}/schema",
    response_model=dict,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_workspace_schema(
        http_request: Request,
        workspace_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Get workspace schema including relationship types and memory subtypes.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Schema with relationship types, memory subtypes, and customization capability

    Raises:
        HTTPException: If workspace not found or schema retrieval fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "read",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.debug("Getting schema for workspace: %s", workspace_id)

        # Verify workspace exists
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        # Get ontology service
        ontology_service = _get_ontology_service()

        # Get relationship types from ontology
        relationship_types = ontology_service.list_relationship_types(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id
        )

        # Get memory subtypes from model
        from ...models.memory import MemorySubtype
        memory_subtypes = [subtype.value for subtype in MemorySubtype]

        return {
            "relationship_types": relationship_types,
            "memory_subtypes": memory_subtypes,
            "can_customize": False,  # OSS: No custom ontologies
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get schema for workspace %s: %s",
            workspace_id,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace schema"
        )


def _serialize_memory(m: dict) -> dict:
    """Convert memory dict to export format."""
    return {
        "id": m["id"],
        "content": m["content"],
        "content_hash": m.get("content_hash", ""),
        "type": m["type"],
        "subtype": m.get("subtype"),
        "importance": m.get("importance", 0.5),
        "tags": m.get("tags", []),
        "metadata": m.get("metadata", {}),
        "abstract": m.get("abstract"),
        "overview": m.get("overview"),
        "session_id": m.get("session_id"),
        "created_at": m.get("created_at"),
        "updated_at": m.get("updated_at"),
    }


async def _generate_export_ndjson(
    storage,
    memory_service,
    workspace_id: str,
    offset: int,
    limit: int,
    include_associations: bool,
    logger: logging.Logger,
):
    """Generate NDJSON export stream."""
    from datetime import datetime, timezone

    # Get workspace stats for counts
    stats = await storage.get_workspace_stats(workspace_id)
    total_memories = stats.get("total_memories", 0)
    total_associations = stats.get("total_associations", 0)

    # Yield header line
    header = {
        "type": "header",
        "version": "1.0",
        "workspace_id": workspace_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_memories": total_memories,
        "total_associations": total_associations,
        "offset": offset,
        "limit": limit,
    }
    yield json.dumps(header, default=str) + "\n"

    # Stream memories in batches
    batch_size = 1000
    batch_offset = offset
    global_index = offset
    memories_exported = 0
    memory_ids = []

    while True:
        # Calculate batch limit
        if limit > 0:
            remaining = limit - memories_exported
            if remaining <= 0:
                break
            batch_limit = min(batch_size, remaining)
        else:
            batch_limit = batch_size

        # Fetch batch
        batch = await storage.get_recent_memories(
            workspace_id,
            created_after=datetime(2000, 1, 1, tzinfo=timezone.utc),
            limit=batch_limit,
            offset=batch_offset,
            detail_level="full",
        )

        if not batch:
            break

        # Stream each memory (convert Memory objects to dicts)
        for m_obj in batch:
            m = m_obj.model_dump() if hasattr(m_obj, 'model_dump') else m_obj
            memory_ids.append(m["id"])
            memory_line = {
                "type": "memory",
                "index": global_index,
                "data": _serialize_memory(m),
            }
            yield json.dumps(memory_line, default=str) + "\n"
            global_index += 1
            memories_exported += 1

        # Stop if we got less than requested (exhausted)
        if len(batch) < batch_limit:
            break

        batch_offset += len(batch)

    # Stream associations if requested
    associations_exported = 0
    if include_associations and memory_ids:
        seen = set()
        for memory_id in memory_ids:
            try:
                assocs = await storage.get_associations(workspace_id, memory_id)
                for a in assocs:
                    key = (a.source_id, a.target_id, a.relationship)
                    if key not in seen:
                        seen.add(key)
                        assoc_line = {
                            "type": "association",
                            "data": {
                                "source_id": a.source_id,
                                "target_id": a.target_id,
                                "relationship_type": a.relationship,
                                "strength": a.strength if hasattr(a, "strength") else 1.0,
                                "metadata": a.metadata if hasattr(a, "metadata") else {},
                            },
                        }
                        yield json.dumps(assoc_line, default=str) + "\n"
                        associations_exported += 1
            except Exception as e:
                logger.warning("Failed to get associations for %s: %s", memory_id, e)

    # Yield footer
    footer = {
        "type": "footer",
        "memories_exported": memories_exported,
        "associations_exported": associations_exported,
    }
    yield json.dumps(footer, default=str) + "\n"


@router.get(
    "/{workspace_id}/export",
    responses={
        200: {"description": "NDJSON stream of workspace data"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def export_workspace(
    http_request: Request,
    workspace_id: str,
    offset: int = 0,
    limit: int = 0,
    include_associations: bool = True,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    memory_service: MemoryService = Depends(get_memory_service),
    logger: logging.Logger = Depends(get_logger),
):
    """Export workspace memories and associations as streaming NDJSON.

    Query params:
    - offset: Skip first N memories (default: 0)
    - limit: Export at most N memories (default: 0 = unlimited)
    - include_associations: Include memory associations (default: true)

    Returns NDJSON stream with lines:
    - {"type":"header", ...}: metadata about export
    - {"type":"memory", "index":N, "data":{...}}: each memory
    - {"type":"association", "data":{...}}: each association
    - {"type":"footer", ...}: summary counts
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "read",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        # Verify workspace exists
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        logger.info("Exporting workspace: %s (offset=%d, limit=%d)", workspace_id, offset, limit)

        return StreamingResponse(
            content=_generate_export_ndjson(
                memory_service.storage,
                memory_service,
                workspace_id,
                offset,
                limit,
                include_associations,
                logger,
            ),
            media_type="application/x-ndjson",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export workspace"
        )


def _process_memory_import(item: MemoryExportItem, workspace_id: str, tenant_id: str, id_mapping: dict) -> tuple[bool, str, str]:
    """Process a single memory import. Returns (success, new_id, error_msg)."""
    from uuid import uuid4
    from ...models.memory import Memory, MemoryType, MemorySubtype

    try:
        # Parse type/subtype
        try:
            mem_type = MemoryType(item.type)
        except (ValueError, KeyError):
            mem_type = MemoryType.EPISODIC

        mem_subtype = None
        if item.subtype:
            try:
                mem_subtype = MemorySubtype(item.subtype)
            except (ValueError, KeyError):
                pass

        # Create new memory with fresh ID
        new_id = f"mem_{uuid4().hex[:16]}"
        id_mapping[item.id] = new_id

        memory = Memory(
            id=new_id,
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            content=item.content,
            content_hash=item.content_hash,
            type=mem_type,
            subtype=mem_subtype,
            importance=item.importance,
            tags=item.tags,
            metadata=item.metadata,
            abstract=item.abstract,
            overview=item.overview,
        )
        return True, new_id, memory
    except Exception as e:
        return False, "", str(e)


def _process_association_import(assoc: AssociationExportItem, id_mapping: dict) -> tuple[bool, dict, str]:
    """Process a single association import. Returns (success, assoc_data, error_msg)."""
    try:
        new_source = id_mapping.get(assoc.source_id)
        new_target = id_mapping.get(assoc.target_id)
        if not new_source or not new_target:
            return False, {}, "Source or target memory not found in mapping"

        from ...models.association import AssociateInput
        assoc_input = AssociateInput(
            source_id=new_source,
            target_id=new_target,
            relationship=assoc.relationship_type,
            strength=assoc.strength,
        )
        return True, assoc_input, ""
    except Exception as e:
        return False, {}, str(e)


@router.post(
    "/{workspace_id}/import",
    response_model=WorkspaceImportResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def import_workspace(
    http_request: Request,
    workspace_id: str,
    request: WorkspaceImportRequest = None,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    memory_service: MemoryService = Depends(get_memory_service),
    logger: logging.Logger = Depends(get_logger),
) -> WorkspaceImportResult:
    """Import memories and associations from JSON or NDJSON export."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "write",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workspace not found: {workspace_id}")

        # Check Content-Type to determine format
        content_type = http_request.headers.get("content-type", "application/json")

        memories_to_import = []
        associations_to_import = []

        if "application/x-ndjson" in content_type:
            # NDJSON format
            body = await http_request.body()
            lines = body.decode().strip().split('\n')

            for line in lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    obj_type = obj.get("type")

                    if obj_type == "memory":
                        data = obj.get("data", {})
                        memories_to_import.append(MemoryExportItem(**data))
                    elif obj_type == "association":
                        data = obj.get("data", {})
                        associations_to_import.append(AssociationExportItem(**data))
                    # Ignore header and footer lines
                except Exception as e:
                    logger.warning("Failed to parse NDJSON line: %s", e)
        else:
            # JSON format (existing behavior)
            if request is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body required for JSON import")
            memories_to_import = request.data.memories
            associations_to_import = request.data.associations

        logger.info("Importing into workspace: %s, memories: %d", workspace_id, len(memories_to_import))

        imported = 0
        skipped = 0
        errors = 0
        details = []
        id_mapping = {}  # old_id -> new_id for association remapping

        for item in memories_to_import:
            try:
                # Check content_hash dedup
                if item.content_hash:
                    existing = await memory_service.storage.get_memory_by_hash(workspace_id, item.content_hash)
                    if existing:
                        id_mapping[item.id] = existing.id
                        skipped += 1
                        details.append(f"Skipped duplicate: {item.id} (hash match: {existing.id})")
                        continue

                # Process memory import
                success, new_id, memory = _process_memory_import(item, workspace_id, ctx.tenant_id, id_mapping)
                if not success:
                    errors += 1
                    details.append(f"Error importing {item.id}: {memory}")
                    logger.warning("Failed to import memory %s: %s", item.id, memory)
                    continue

                await memory_service.storage.create_memory(workspace_id, memory)
                imported += 1

            except Exception as e:
                errors += 1
                details.append(f"Error importing {item.id}: {str(e)}")
                logger.warning("Failed to import memory %s: %s", item.id, e)

        # Import associations with remapped IDs
        assoc_imported = 0
        for assoc in associations_to_import:
            try:
                success, assoc_input, error_msg = _process_association_import(assoc, id_mapping)
                if success:
                    await memory_service.storage.create_association(workspace_id, assoc_input)
                    assoc_imported += 1
                else:
                    logger.debug("Skipped association: %s", error_msg)
            except Exception as e:
                logger.warning("Failed to import association: %s", e)

        if assoc_imported > 0:
            details.append(f"Imported {assoc_imported} associations")

        logger.info(
            "Import complete for workspace %s: imported=%d, skipped=%d, errors=%d",
            workspace_id, imported, skipped, errors
        )

        return WorkspaceImportResult(
            imported=imported,
            skipped_duplicates=skipped,
            errors=errors,
            details=details,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to import into workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to import workspace data")


class WorkspacesAPIPlugin(Plugin):
    """Plugin to register workspaces API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
