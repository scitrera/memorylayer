"""
AsyncIO Task Service implementation.

Simple in-memory task service using asyncio for local development and single-node deployments.
"""
import asyncio
from typing import Callable, Awaitable
from uuid import uuid4
from logging import Logger

from scitrera_app_framework import get_logger, Variables, ext_parse_bool

from .base import TaskService, TaskServicePluginBase, TaskStatus

MEMORYLAYER_TASKS_ENABLED = 'MEMORYLAYER_TASKS_ENABLED'
DEFAULT_TASKS_ENABLED = True


class AsyncIOTaskService(TaskService):
    """
    Simple in-memory task service using asyncio.

    Features:
    - No persistence (tasks lost on restart)
    - Single-process only
    - Perfect for local dev / single-node OSS

    """

    def __init__(self, v: Variables = None, tasks_enabled: bool = DEFAULT_TASKS_ENABLED):
        """
        Initialize the service.

        Args:
            v: Variables for logging context
        """
        self._tasks_enabled = tasks_enabled
        self._tasks: dict[str, asyncio.Task] = {}
        self._recurring: dict[str, bool] = {}
        self._handlers: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized AsyncIOTaskService")

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
            task_type: Type of task to execute
            payload: Task payload data
            delay_seconds: Delay before execution
            priority: Task priority (ignored in asyncio implementation)

        Returns:
            Task ID
        """
        if not self._tasks_enabled:
            self.logger.debug("Tasks are disabled, skipping schedule_task for type: %s", task_type)
            return None
        task_id = f"task_{uuid4().hex[:12]}"  # TODO: use utils id generator function instead of reinvent?

        async def run_after_delay():
            try:
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

                handler = self._handlers.get(task_type)
                if handler:
                    self.logger.debug("Executing task %s (type: %s)", task_id, task_type)
                    await handler(payload)
                    self.logger.debug("Task %s completed", task_id)
                else:
                    self.logger.error("No handler registered for task type: %s", task_type)
            except Exception as e:
                self.logger.error("Task %s failed: %s", task_id, e, exc_info=True)

        self._tasks[task_id] = asyncio.create_task(run_after_delay())
        return task_id

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
            Schedule ID
        """
        if not self._tasks_enabled:
            self.logger.debug("Tasks are disabled, skipping schedule_task for type: %s", task_type)
            return None
        schedule_id = f"sched_{uuid4().hex[:12]}"  # TODO: use utils id generator function instead of reinvent?
        self._recurring[schedule_id] = True

        async def run_recurring():
            while self._recurring.get(schedule_id):
                handler = self._handlers.get(task_type)
                if handler:
                    try:
                        self.logger.debug("Executing recurring task (type: %s)", task_type)
                        await handler(payload)
                    except Exception as e:
                        self.logger.error(
                            "Recurring task %s failed: %s",
                            task_type,
                            e,
                            exc_info=True
                        )
                else:
                    self.logger.error("No handler registered for task type: %s", task_type)

                await asyncio.sleep(interval_seconds)

        asyncio.create_task(run_recurring())  # TODO: why don't we keep a ref of schedule_id --> task ??
        self.logger.info(
            "Scheduled recurring task %s: type=%s, interval=%ss",
            schedule_id,
            task_type,
            interval_seconds
        )
        return schedule_id

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task or recurring schedule.

        Args:
            task_id: Task or schedule ID to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        # Check if it's a one-time task
        task = self._tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            self.logger.info("Cancelled task %s", task_id)
            return True

        # Check if it's a recurring schedule
        if task_id in self._recurring:
            self._recurring[task_id] = False
            self.logger.info("Cancelled recurring schedule %s", task_id)
            return True

        return False

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Get task execution status.

        Args:
            task_id: Task ID to check

        Returns:
            Current task status
        """
        task = self._tasks.get(task_id)
        if not task:
            # Check if it's a recurring schedule
            if task_id in self._recurring:
                return TaskStatus.RUNNING if self._recurring[task_id] else TaskStatus.CANCELLED
            return TaskStatus.NOT_FOUND

        if task.done():
            if task.cancelled():
                return TaskStatus.CANCELLED
            elif task.exception():
                return TaskStatus.FAILED
            else:
                return TaskStatus.COMPLETED

        return TaskStatus.RUNNING

    def register_handler(
            self,
            task_type: str,
            handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Register a handler for a task type.

        Args:
            task_type: Task type identifier
            handler: Async handler function
        """
        self._handlers[task_type] = handler
        self.logger.info("Registered handler for task type: %s", task_type)


class AsyncIOTaskServicePlugin(TaskServicePluginBase):
    """
    Plugin that creates and manages the AsyncIOTaskService instance.

    NOTE: This is the PLUGIN only - it handles lifecycle.
    The service implementation is in AsyncIOTaskService.
    """

    # This MUST match what users set in MEMORYLAYER_TASK_PROVIDER
    PROVIDER_NAME = 'asyncio'

    def initialize(self, v: Variables, logger: Logger) -> TaskService:
        """
        Create and return the service instance.

        This method is called by the plugin framework during startup.

        Args:
            v: Variables for configuration and logging
            logger: Logger instance

        Returns:
            Initialized TaskService instance
        """
        tasks_enabled = v.environ(MEMORYLAYER_TASKS_ENABLED, DEFAULT_TASKS_ENABLED, type_fn=ext_parse_bool)
        return AsyncIOTaskService(
            v=v,
            tasks_enabled=tasks_enabled,
        )
