"""
Task Handler Plugin Base.

Base class for task handler plugins that are auto-discovered via multi-extension.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from logging import Logger

from scitrera_app_framework import Plugin, Variables, ext_parse_bool, get_extensions

from ..memory import EXT_MEMORY_SERVICE
from ..session import EXT_SESSION_SERVICE
from .aether import (
    DEFAULT_MEMORYLAYER_TASKS_INPROCESS_WORKER,
    MEMORYLAYER_TASKS_INPROCESS_WORKER,
)
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

        # Only register handlers when the server acts as an in-process worker.
        # When disabled (prod split-role), the server still schedules/produces
        # tasks but does not register handlers (and does not claim — see
        # AetherTaskService.bind_client); dedicated WorkerRunner processes
        # consume the "memorylayer" POOL instead.
        inprocess_worker = v.environ(
            MEMORYLAYER_TASKS_INPROCESS_WORKER,
            DEFAULT_MEMORYLAYER_TASKS_INPROCESS_WORKER,
            type_fn=ext_parse_bool,
        )
        if not inprocess_worker:
            logger.info(
                "In-process worker disabled (%s=false): skipping task handler registration "
                "(run dedicated workers instead)",
                MEMORYLAYER_TASKS_INPROCESS_WORKER,
            )
            return task_service

        # Register task service handlers
        for handler_plugin in get_extensions(EXT_MULTI_TASK_HANDLERS, v).values():  # type: TaskHandlerPlugin
            # The multi-extension collection also carries this setup plugin's own
            # return value (the TaskService), which is not a handler — skip it so
            # a non-handler entry can't abort the loop mid-iteration (leaving
            # later handlers unregistered depending on init/insertion order).
            if not isinstance(handler_plugin, TaskHandlerPlugin):
                continue
            task_type: str = handler_plugin.get_task_type()
            handler: Callable[[Variables, dict], Awaitable[None]] = handler_plugin.handle
            task_service.register_handler(task_type, handler)

        return task_service  # not really **critial** but we can pass through the task_service for convenience

    async def async_ready(self, v: Variables, logger: Logger, value: TaskService) -> None:
        task_service: TaskService = value
        logger.info("Scheduling Recurring Task Handlers")
        for handler_plugin in get_extensions(EXT_MULTI_TASK_HANDLERS, v).values():  # type: TaskHandlerPlugin
            # Skip non-handler entries (e.g. this setup plugin's own TaskService
            # return value, which lacks get_schedule): a bare AttributeError here
            # would abort scheduling for every handler ordered after it.
            if not isinstance(handler_plugin, TaskHandlerPlugin):
                continue
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
