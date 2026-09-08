"""Shared HTTP mechanics for domain-specific revisioned-resource APIs."""

import logging
from typing import NoReturn

from fastapi import HTTPException, Request, status

from ...models.versioned_resource import (
    VersionedResourceConflictError,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
)


def required_header(request: Request, name: str) -> str:
    value = request.headers.get(name, "").strip()
    if not value:
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail=f"{name} header is required")
    if len(value) > 200:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{name} header is too long")
    return value


def actor(ctx) -> str | None:
    if ctx.actor is not None:
        return f"{ctx.actor.type}:{ctx.actor.id}"
    return ctx.effective_subject_id()


def raise_api_error(exc: Exception, operation: str, logger_name: str) -> NoReturn:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, VersionedResourceNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, VersionedResourceConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, VersionedResourcePreconditionFailedError):
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail=str(exc))
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    logging.getLogger(logger_name).error("Failed to %s: %s", operation, exc, exc_info=True)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to {operation}")
