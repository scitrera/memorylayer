"""Task service package."""
from .base import (
    TaskServicePluginBase,
    EXT_TASK_SERVICE,
    EXT_MULTI_TASK_HANDLERS,
    TaskService,
    TaskStatus,
    TaskSchedule,
)
from .handlers import TaskHandlerPlugin

from scitrera_app_framework import Variables, get_extension


def get_task_service(v: Variables = None) -> TaskService:
    """Get the configured TaskService instance."""
    return get_extension(EXT_TASK_SERVICE, v)


__all__ = (
    'TaskService',
    'TaskServicePluginBase',
    'TaskHandlerPlugin',
    'TaskStatus',
    'TaskSchedule',
    'get_task_service',
    'EXT_TASK_SERVICE',
    'EXT_MULTI_TASK_HANDLERS',
)
