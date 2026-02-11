"""
Task Service - Base classes and protocols.

Provides background task scheduling abstraction for memory lifecycle operations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Awaitable, Optional
from logging import Logger

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_TASK_PROVIDER, DEFAULT_MEMORYLAYER_TASK_PROVIDER

# Extension point constants
EXT_TASK_SERVICE = 'memorylayer-task-service'
EXT_MULTI_TASK_HANDLERS = 'memorylayer-multi-task-handlers'


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
    async def schedule_task(
        self,
        task_type: str,
        payload: dict,
        delay_seconds: int = 0,
        priority: int = 5
    ) -> str:
        """
        Schedule a task for background execution.

        Args:
            task_type: Type of task to execute (matches registered handler)
            payload: Task payload data
            delay_seconds: Delay before execution (default: immediate)
            priority: Task priority (1-10, lower is higher priority)

        Returns:
            Task ID for tracking
        """
        pass

    @abstractmethod
    async def schedule_recurring(
        self,
        task_type: str,
        interval_seconds: int,
        payload: dict
    ) -> str:
        """
        Schedule a recurring task.

        Args:
            task_type: Type of task to execute
            interval_seconds: Interval between executions
            payload: Task payload data

        Returns:
            Schedule ID for tracking/cancellation
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
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Register a handler for a task type.

        Called by lifecycle plugin during startup to register all task handlers.

        Args:
            task_type: Task type identifier
            handler: Async handler function that processes the task payload
        """
        pass


# noinspection PyAbstractClass
class TaskServicePluginBase(Plugin):
    """
    Base plugin for TaskService implementations.

    Subclasses MUST:
    1. Set PROVIDER_NAME to their provider name (e.g., 'asyncio', 'distributed')
    2. Implement initialize() to return a TaskService instance
    """

    # Subclasses MUST set this to their provider name
    PROVIDER_NAME: str = None

    def name(self) -> str:
        """Unique plugin name combining extension point and provider."""
        return f"{EXT_TASK_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        """Return the extension point this plugin provides."""
        return EXT_TASK_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        """Check if this provider is enabled based on config."""
        return enabled_option_pattern(
            self, v,
            MEMORYLAYER_TASK_PROVIDER,
            self_attr='PROVIDER_NAME'
        )

    def on_registration(self, v: Variables) -> None:
        """Set default value when plugin is registered."""
        v.set_default_value(
            MEMORYLAYER_TASK_PROVIDER,
            DEFAULT_MEMORYLAYER_TASK_PROVIDER
        )

    def get_dependencies(self, v: Variables):
        """Declare dependencies. Override in subclasses if needed."""
        return ()
