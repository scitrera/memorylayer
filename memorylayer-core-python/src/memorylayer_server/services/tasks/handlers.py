"""
Task Handler Plugin Base.

Base class for task handler plugins that are auto-discovered via multi-extension.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from logging import Logger

from scitrera_app_framework import Plugin, Variables, get_extensions

from ..memory import EXT_MEMORY_SERVICE
from ..session import EXT_SESSION_SERVICE
from .base import EXT_MULTI_TASK_HANDLERS, EXT_TASK_SERVICE, TaskSchedule, TaskService


class TaskHandlerPlugin(Plugin, ABC):
    """
    Base class for task handler plugins.

    Task handlers are auto-discovered via the EXT_MULTI_TASK_HANDLERS extension point
    and registered with the TaskService during startup.

    Subclasses must implement:
    - get_task_type(): Return the task type this handler processes
    - handle(payload): Execute the task
    - get_schedule(): Return recurring schedule or None
    """

    @abstractmethod
    def get_task_type(self) -> str:
        """
        Return the task type this handler processes.

        This is used to route tasks to the correct handler.

        Returns:
            Task type identifier (e.g., "decay_memories", "detect_open_threads")
        """
        pass

    @abstractmethod
    async def handle(self, v: Variables, payload: dict) -> None:
        """
        Execute the task with given payload.

        Args:
            payload: Task-specific data

        Raises:
            Exception: Task execution errors are logged but not re-raised
        """
        pass

    @abstractmethod
    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        """
        Return a recurring schedule, or None if not recurring. Takes variables instance for context
        for dynamic schedules that are dependent on configuration.

        Returns:
            TaskSchedule with interval and default payload, or None for one-time tasks
        """
        pass

    def initialize(self, v, logger) -> object | None:
        return self  # use the plugin instance as the handler instance

    def extension_point_name(self, v: Variables) -> str:
        """Return the multi-extension point for task handlers."""
        return EXT_MULTI_TASK_HANDLERS

    def is_enabled(self, v: Variables) -> bool:
        """Disable 'single' extension for multi-extension plugins."""
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        """Mark this as a multi-extension plugin."""
        return True


class TaskHandlersSetupPlugin(Plugin):
    """
    Configure task handlers for task service
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_TASK_HANDLERS

    def initialize(self, v, logger) -> object | None:
        logger.info("Initializing Task Service Handlers")
        task_service: TaskService = self.get_extension(EXT_TASK_SERVICE, v)

        # Register task service handlers
        for handler_plugin in get_extensions(EXT_MULTI_TASK_HANDLERS, v).values():  # type: TaskHandlerPlugin
            task_type: str = handler_plugin.get_task_type()
            handler: Callable[[Variables, dict], Awaitable[None]] = handler_plugin.handle
            task_service.register_handler(task_type, handler)

        return task_service  # not really **critial** but we can pass through the task_service for convenience

    async def async_ready(self, v: Variables, logger: Logger, value: TaskService) -> None:
        task_service: TaskService = value
        logger.info("Scheduling Recurring Task Handlers")
        for handler_plugin in get_extensions(EXT_MULTI_TASK_HANDLERS, v).values():  # type: TaskHandlerPlugin
            # Schedule recurring tasks
            schedule = handler_plugin.get_schedule(v)
            if schedule:
                # Guard: verify payload is serializable (catches service objects at startup)
                import json

                try:
                    json.dumps(schedule.default_payload)
                except TypeError as e:
                    raise TypeError(
                        "Task handler '%s' has a non-serializable default_payload: %s. "
                        "Move service resolution from get_schedule() to handle()." % (handler_plugin.get_task_type(), e)
                    ) from e

                await task_service.schedule_recurring(handler_plugin.get_task_type(), schedule.interval_seconds, schedule.default_payload)

        return

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (
            EXT_TASK_SERVICE,  # register handlers must come after task service initialization
            EXT_MEMORY_SERVICE,  # ensure we have memory service available before registering tasks
            EXT_SESSION_SERVICE,  # ensure we have session service available before registering tasks
        )
