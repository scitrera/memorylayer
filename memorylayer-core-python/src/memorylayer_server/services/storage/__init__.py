from scitrera_app_framework import Variables, get_extension

from .base import EXT_STORAGE_BACKEND, StorageBackend


def get_storage_backend(v: Variables = None) -> StorageBackend:
    return get_extension(EXT_STORAGE_BACKEND, v)


__all__ = (
    "StorageBackend",
    "get_storage_backend",
    "EXT_STORAGE_BACKEND",
)
