"""Skills service package."""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_SKILLS_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base
from ...config import DEFAULT_MEMORYLAYER_SKILLS_PROVIDER, MEMORYLAYER_SKILLS_PROVIDER
from .base import SkillsService

SkillsServicePluginBase = make_service_plugin_base(
    ext_name=EXT_SKILLS_SERVICE,
    config_key=MEMORYLAYER_SKILLS_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_SKILLS_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)


def get_skills_service(v: Variables = None) -> SkillsService:
    """Get the skills service instance."""
    return get_extension(EXT_SKILLS_SERVICE, v)


__all__ = (
    "SkillsService",
    "SkillsServicePluginBase",
    "get_skills_service",
    "EXT_SKILLS_SERVICE",
)
