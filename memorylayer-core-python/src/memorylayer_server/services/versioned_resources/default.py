"""Default versioned-resource service plugin."""

import logging

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import VersionedResourceServicePluginBase
from .base import VersionedResourceService


class DefaultVersionedResourceServicePlugin(VersionedResourceServicePluginBase):
    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> VersionedResourceService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return VersionedResourceService(storage)
