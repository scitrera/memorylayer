"""
Task Service - Base classes and protocols.

Provides background task scheduling abstraction for memory lifecycle operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from scitrera_app_framework.api import Variables

from ...config import DEFAULT_MEMORYLAYER_TASK_PROVIDER, MEMORYLAYER_TASK_PROVIDER
from .._constants import (
    EXT_MULTI_TASK_HANDLERS,  # noqa: F401 — re-exported for handlers.py and __init__.py
    EXT_STORAGE_BACKEND,
    EXT_TASK_SERVICE,
)
from .._plugin_factory import make_service_plugin_base


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_FOUND = "not_found"


@dataclass
class TaskSchedule:
    """Configuration for recurring task schedule."""

    interval_seconds: int
    default_payload: dict


class TaskService(ABC):
    """
    Interface for background task management.

    NOTE: This is a pure ABC - no plugin inheritance here.
    Plugin lifecycle is handled by TaskServicePluginBase.
    """

    @abstractmethod
    async def schedule_task(self, task_type: str, payload: dict, delay_seconds: int = 0, priority: int = 5) -> str | None:
        """
        Schedule a task for background execution.

        Args:
            task_type: Type of task to execute (matches registered handler)
            payload: Task payload data
            delay_seconds: Delay before execution (default: immediate)
            priority: Task priority (1-10, lower is higher priority)

        Returns:
            Task ID for tracking, or None if tasks are disabled.
        """
        pass

    @abstractmethod
    async def schedule_recurring(self, task_type: str, interval_seconds: int, payload: dict) -> str | None:
        """
        Schedule a recurring task.

        Args:
            task_type: Type of task to execute
            interval_seconds: Interval between executions
            payload: Task payload data

        Returns:
            Schedule ID for tracking/cancellation, or None if tasks are disabled.
        """
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or recurring task.

        Args:
            task_id: Task or schedule ID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Get task execution status.

        Args:
            task_id: Task ID to check

        Returns:
            Current task status
        """
        pass

    @abstractmethod
    def register_handler(self, task_type: str, handler: Callable[[Variables, dict], Awaitable[None]]) -> None:
        """
        Register a handler for a task type.

        Called by lifecycle plugin during startup to register all task handlers.

        Args:
            task_type: Task type identifier
            handler: Async handler function that processes the task payload
        """
        pass


# noinspection PyAbstractClass
TaskServicePluginBase = make_service_plugin_base(
    ext_name=EXT_TASK_SERVICE,
    config_key=MEMORYLAYER_TASK_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_TASK_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)
