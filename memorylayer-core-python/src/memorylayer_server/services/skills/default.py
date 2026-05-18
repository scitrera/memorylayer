"""Default SkillsService plugin implementation."""

import logging

from scitrera_app_framework import Variables, get_extension, get_logger

from .._constants import EXT_SKILLS_SERVICE, EXT_STORAGE_BACKEND
from ..storage import StorageBackend
from . import SkillsServicePluginBase
from .base import SkillsService


class DefaultSkillsServicePlugin(SkillsServicePluginBase):
    """Plugin for the default SkillsService backed by StorageBackend."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> SkillsService:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        return SkillsService(storage=storage)
