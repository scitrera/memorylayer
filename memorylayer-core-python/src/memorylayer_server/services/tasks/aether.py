"""
Aether-backed distributed task service for MemoryLayer.

This module implements the ``TaskService`` ABC using the shared
``AsyncServiceClient`` owned by :class:`AetherServiceConnection` to
distribute background tasks across worker processes via Aether's messaging
system.

Architecture
------------
* The MemoryLayer server connects to Aether through a single Service
  identity (``sv::memorylayer::*``) managed by ``AetherServiceConnection``.
  This task service obtains the shared client rather than creating its own
  connection.
* ``schedule_task`` calls Aether's ``create_task()`` API which provides
  PostgreSQL-backed persistence, retry policies, and a DLQ.  Workers receive
  tasks via ``on_task_assignment`` callbacks using POOL assignment mode
  (competing consumers).
* ``schedule_recurring`` registers schedules in Aether's workflow engine
  with deterministic IDs for idempotent multi-instance registration.
* ``cancel_task`` deletes Aether workflow schedules or cancels local tasks.

Configuration (environment variables)
-------------------------------------
``AETHER_TASKS_ENABLED``
    Master enable/disable toggle (default ``true``).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from logging import Logger
from typing import Callable, Awaitable, Optional
from uuid import uuid4

from scitrera_rt_data.serialization.msgpack import msgpack_serialize, msgpack_deserialize

from scitrera_app_framework import get_logger, get_extension, Variables, ext_parse_bool

from memorylayer_server.services.tasks.base import (
    TaskService,
    TaskServicePluginBase,
    TaskStatus,
    EXT_STORAGE_BACKEND,
)

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION

# ---------------------------------------------------------------------------
# Payload deserialization
# ---------------------------------------------------------------------------


def _deserialize_task_payload(data: bytes) -> Optional[dict]:
    """Deserialize a task payload, trying msgpack first then JSON.

    Tasks created by the ML server use msgpack; tasks created by the
    workflow engine's schedule actions use JSON.  The gateway passes
    the payload through as opaque bytes.
    """
    # Try msgpack first (from ML server _create_task)
    try:
        return msgpack_deserialize(data)
    except Exception:
        pass
    # Try JSON (from workflow engine schedule actions)
    try:
        return json.loads(data)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
AETHER_TASKS_ENABLED = "AETHER_TASKS_ENABLED"
DEFAULT_AETHER_TASKS_ENABLED = True

# Implementation name used for task targeting (self)
_TASK_IMPLEMENTATION = "memorylayer"


def _build_task_envelope(
        task_id: str,
        task_type: str,
        payload: dict,
        priority: int = 5,
        auth_context: Optional[dict] = None,
) -> dict:
    """Build the canonical JSON envelope for an Aether task message.

    Returns a plain ``dict`` ready for ``json.dumps`` serialisation.
    """
    return {
        "task_id": task_id,
        "task_type": task_type,
        "payload": payload,
        "auth_context": auth_context or {},
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
        "priority": priority,
    }


class AetherTaskService(TaskService):
    """Distributed task service backed by the shared Aether client.

    Uses the ``AsyncServiceClient`` from :class:`AetherServiceConnection` instead
    of maintaining its own connection.

    The service operates in two phases:

    1. **Construction** (``__init__``): No I/O; stores configuration and
       creates internal state structures.
    2. **Binding** (``bind_client``): Obtains the shared client from
       AetherServiceConnection and registers the task assignment handler.

    Shutdown is performed via ``disconnect`` which cancels all recurring
    tasks (the shared client lifecycle is managed by AetherServiceConnection).
    """

    def __init__(
            self,
            v: Variables,
            *,
            tasks_enabled: bool = DEFAULT_AETHER_TASKS_ENABLED,
    ) -> None:
        self._v = v
        self._tasks_enabled = tasks_enabled

        # Handlers registered via register_handler (task_type -> handler)
        self._handlers: dict[str, Callable[[Variables, dict], Awaitable[None]]] = {}

        # Tracking state for scheduled tasks
        self._task_status: dict[str, TaskStatus] = {}

        # Recurring schedule state: schedule_id -> active flag
        self._recurring: dict[str, bool] = {}
        # Recurring schedule asyncio tasks: schedule_id -> asyncio.Task
        self._recurring_tasks: dict[str, asyncio.Task] = {}

        # Shared Aether client (set during bind_client)
        self._client = None  # type: ignore[assignment]
        # Workspace from the agent service
        self._workspace: str = "_system"

        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized AetherTaskService (tasks_enabled=%s)", tasks_enabled)

    # ------------------------------------------------------------------
    # Client binding (replaces own connect/disconnect lifecycle)
    # ------------------------------------------------------------------

    def bind_client(self, agent_service) -> None:
        """Bind to the shared client from AetherServiceConnection.

        Also registers this service's task assignment handler on the
        agent service so incoming task assignments are dispatched here.
        """
        self._client = agent_service.client
        self._workspace = agent_service.workspace
        agent_service.set_task_assignment_handler(self._handle_task_assignment)
        self.logger.info(
            "Bound to shared Aether client (workspace=%s, connected=%s)",
            self._workspace,
            self._client is not None,
        )

    async def disconnect(self) -> None:
        """Clean up local state.

        The shared client lifecycle is managed by AetherServiceConnection,
        so we only cancel legacy asyncio-based recurring tasks here.
        """
        for schedule_id in list(self._recurring):
            await self.cancel_task(schedule_id)
        # Release reference (do NOT close — not our client)
        self._client = None

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the shared Aether client is available."""
        return self._client is not None

    # ------------------------------------------------------------------
    # TaskService ABC implementation
    # ------------------------------------------------------------------

    async def schedule_task(
            self,
            task_type: str,
            payload: dict,
            delay_seconds: int = 0,
            priority: int = 5,
    ) -> str:
        """Schedule a one-shot task via Aether's task lifecycle.

        Args:
            task_type: Type of task to execute (matches registered handler).
            payload: Task payload data.
            delay_seconds: Delay before creating (default: immediate).
            priority: Task priority 1-10, lower is higher (default: 5).

        Returns:
            Unique task ID (``atask_<hex>`` prefix), or empty string if
            tasks are disabled.
        """
        if not self._tasks_enabled:
            self.logger.debug("Tasks disabled, skipping schedule_task for type: %s", task_type)
            return ""

        task_id = f"atask_{uuid4().hex[:12]}"
        self._task_status[task_id] = TaskStatus.PENDING

        if delay_seconds > 0:
            asyncio.create_task(self._delayed_create(task_id, task_type, payload, delay_seconds))
        else:
            await self._create_task(task_id, task_type, payload)

        return task_id

    async def schedule_recurring(
            self,
            task_type: str,
            interval_seconds: int,
            payload: dict,
    ) -> str:
        """Register a recurring task schedule via Aether's workflow engine.

        Uses upsert_schedule with a deterministic ID so multiple server
        instances register the same schedule idempotently.

        Args:
            task_type: Type of task to execute.
            interval_seconds: Seconds between executions.
            payload: Task payload data.

        Returns:
            Schedule ID (``ml-{task_type}``), or empty string if disabled.
        """
        if not self._tasks_enabled:
            self.logger.debug("Tasks disabled, skipping schedule_recurring for type: %s", task_type)
            return ""

        schedule_id = f"ml-{task_type}"

        if self._client is None:
            self.logger.error(
                "Cannot register recurring schedule %s: not connected to Aether gateway",
                schedule_id,
            )
            return ""

        action = {
            "type": "create_task",
            "task_type": f"memorylayer-task.{task_type}",
            "target_implementation": _TASK_IMPLEMENTATION,
            "workspace": self._workspace,
            "payload": payload,
        }

        try:
            resp = await self._client.upsert_schedule(
                schedule_id=schedule_id,
                name=f"memorylayer:{task_type}",
                schedule_type="interval",
                schedule_expr=f"{interval_seconds}s",
                action=action,
                workspace=self._workspace,
                miss_policy="fire_once",
                max_concurrent=1,
            )
            if resp and resp.success:
                self.logger.info(
                    "Registered recurring schedule %s: type=%s, interval=%ss",
                    schedule_id,
                    task_type,
                    interval_seconds,
                )
            elif resp:
                self.logger.error(
                    "Failed to register schedule %s: %s",
                    schedule_id,
                    resp.error,
                )
            else:
                self.logger.error(
                    "Timeout registering schedule %s",
                    schedule_id,
                )
        except Exception:
            self.logger.error(
                "Failed to register recurring schedule %s",
                schedule_id,
                exc_info=True,
            )

        return schedule_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or recurring task.

        For recurring schedules (IDs starting with ``ml-``), deletes the
        schedule from Aether's workflow engine.  For legacy asyncio-based
        recurring schedules, stops the local loop.  For one-shot tasks,
        this is best-effort.

        Args:
            task_id: Task ID or schedule ID to cancel.

        Returns:
            ``True`` if cancelled, ``False`` if not found.
        """
        # Aether workflow schedule (new-style deterministic IDs)
        if task_id.startswith("ml-") and self._client is not None:
            try:
                resp = await self._client.delete_schedule(task_id)
                if resp and resp.success:
                    self.logger.info("Deleted Aether schedule %s", task_id)
                    return True
                self.logger.warning("Failed to delete Aether schedule %s: %s", task_id, resp.error if resp else "timeout")
            except Exception:
                self.logger.error("Error deleting Aether schedule %s", task_id, exc_info=True)
            return False

        # Legacy asyncio recurring schedules
        if task_id in self._recurring:
            self._recurring[task_id] = False
            self._task_status[task_id] = TaskStatus.CANCELLED

            recurring_task = self._recurring_tasks.pop(task_id, None)
            if recurring_task is not None and not recurring_task.done():
                recurring_task.cancel()
                try:
                    await recurring_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("Cancelled recurring schedule %s", task_id)
            return True

        # One-shot tasks: mark as cancelled locally (best-effort)
        if task_id in self._task_status:
            current = self._task_status[task_id]
            if current == TaskStatus.PENDING:
                self._task_status[task_id] = TaskStatus.CANCELLED
                self.logger.info("Cancelled pending task %s", task_id)
                return True

        return False

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get the status of a task or recurring schedule."""
        status = self._task_status.get(task_id)
        if status is not None:
            return status

        # Check recurring dict as fallback
        if task_id in self._recurring:
            return TaskStatus.RUNNING if self._recurring[task_id] else TaskStatus.CANCELLED

        return TaskStatus.NOT_FOUND

    def register_handler(
            self,
            task_type: str,
            handler: Callable[[Variables, dict], Awaitable[None]],
    ) -> None:
        """Register a handler for a task type."""
        self._handlers[task_type] = handler
        self.logger.debug("Registered handler for task type: %s", task_type)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _create_task(self, task_id: str, task_type: str, payload: dict) -> None:
        """Create a task via Aether's task lifecycle (create_task).

        Uses POOL assignment mode so that competing worker processes receive
        task assignments via their ``on_task_assignment`` callback.
        """
        if self._client is None:
            self.logger.error(
                "Cannot create aether task %s: not connected to Aether gateway",
                task_id,
            )
            self._task_status[task_id] = TaskStatus.FAILED
            return

        from scitrera_aether_client import POOL

        aether_task_type = f"memorylayer-task.{task_type}"
        try:
            await self._client.create_task(
                task_type=aether_task_type,
                workspace=self._workspace,
                metadata={"task_id": task_id},
                payload=msgpack_serialize(payload),
                target_implementation=_TASK_IMPLEMENTATION,
                assignment_mode=POOL,
            )
            self._task_status[task_id] = TaskStatus.RUNNING
            self.logger.debug(
                "Created aether task %s (aether_type=%s)",
                task_id,
                aether_task_type,
            )
        except Exception:
            self._task_status[task_id] = TaskStatus.FAILED
            self.logger.error(
                "Failed to create aether task %s (aether_type=%s)",
                task_id,
                aether_task_type,
                exc_info=True,
            )

    async def _delayed_create(
            self,
            task_id: str,
            task_type: str,
            payload: dict,
            delay_seconds: int,
    ) -> None:
        """Wait *delay_seconds* then create the aether task."""
        try:
            await asyncio.sleep(delay_seconds)
            await self._create_task(task_id, task_type, payload)
        except asyncio.CancelledError:
            self._task_status[task_id] = TaskStatus.CANCELLED
        except Exception:
            self._task_status[task_id] = TaskStatus.FAILED
            self.logger.error(
                "Delayed publish for task %s failed",
                task_id,
                exc_info=True,
            )

    async def _handle_task_assignment(self, assignment) -> None:
        """Process a native Aether task assignment received via POOL routing.

        Deserializes the payload from the native binary ``assignment.payload``
        (msgpack) with a fallback to the legacy JSON ``assignment.metadata["payload"]``
        for backward compatibility.

        After execution, reports task completion or failure back to Aether.
        """
        raw_task_type: str = assignment.task_type or ""
        prefix = "memorylayer-task."
        task_type = raw_task_type[len(prefix):] if raw_task_type.startswith(prefix) else raw_task_type

        aether_task_id: str = assignment.task_id or ""
        ml_task_id: str = assignment.metadata.get("task_id", aether_task_id or "<unknown>")

        handler = self._handlers.get(task_type)
        if handler is None:
            self.logger.warning(
                "No handler registered for task type %s (task_id=%s), failing task",
                task_type,
                ml_task_id,
            )
            await self._report_task_failed(aether_task_id, f"no handler for task type: {task_type}")
            return

        # Deserialize payload: try msgpack first (from ML server _create_task),
        # then JSON (from workflow engine schedule actions), then legacy metadata.
        if assignment.payload:
            task_payload = _deserialize_task_payload(assignment.payload)
            if task_payload is None:
                self.logger.error(
                    "Failed to deserialize payload for task %s (tried msgpack and JSON)",
                    ml_task_id,
                )
                await self._report_task_failed(aether_task_id, "payload deserialization error")
                return
        else:
            raw_payload = assignment.metadata.get("payload", "{}")
            try:
                task_payload = json.loads(raw_payload)
            except json.JSONDecodeError as exc:
                self.logger.error(
                    "Invalid JSON payload in task assignment %s: %s",
                    ml_task_id,
                    exc,
                )
                await self._report_task_failed(aether_task_id, f"JSON decode error: {exc}")
                return

        self.logger.info("Executing task %s (type=%s)", ml_task_id, task_type)
        try:
            await handler(self._v, task_payload)
            self._task_status[ml_task_id] = TaskStatus.COMPLETED
            self.logger.info("Task %s completed successfully", ml_task_id)
            await self._report_task_completed(aether_task_id)
        except Exception as exc:
            self._task_status[ml_task_id] = TaskStatus.FAILED
            self.logger.error("Task %s failed", ml_task_id, exc_info=True)
            await self._report_task_failed(aether_task_id, str(exc))

    async def _report_task_completed(self, aether_task_id: str) -> None:
        """Report task completion to Aether. Best-effort; failures are logged."""
        if not aether_task_id or self._client is None:
            return
        try:
            await self._client.complete_task(aether_task_id, timeout=5.0)
        except Exception:
            self.logger.warning(
                "Failed to report task %s as completed to Aether",
                aether_task_id,
                exc_info=True,
            )

    async def _report_task_failed(self, aether_task_id: str, reason: str = "") -> None:
        """Report task failure to Aether. Best-effort; failures are logged."""
        if not aether_task_id or self._client is None:
            return
        try:
            await self._client.fail_task(aether_task_id, reason=reason, timeout=5.0)
        except Exception:
            self.logger.warning(
                "Failed to report task %s as failed to Aether",
                aether_task_id,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class AetherTaskServicePlugin(TaskServicePluginBase):
    """Plugin that creates and manages an :class:`AetherTaskService` instance.

    Enabled when ``MEMORYLAYER_TASK_PROVIDER=aether``.

    Lifecycle:
        ``initialize`` -- constructs the service (no I/O).
        ``async_ready`` -- binds to the shared Aether client from AetherServiceConnection.
        ``async_stopping`` -- disconnects and cancels recurring tasks.
    """

    PROVIDER_NAME = "aether"

    def initialize(self, v: Variables, logger: Logger) -> AetherTaskService:
        """Create and return the service instance (no I/O)."""
        tasks_enabled = v.environ(AETHER_TASKS_ENABLED, DEFAULT_AETHER_TASKS_ENABLED, type_fn=ext_parse_bool)

        return AetherTaskService(
            v,
            tasks_enabled=tasks_enabled,
        )

    async def async_ready(self, v: Variables, logger: Logger, value: AetherTaskService) -> None:
        """Bind the task service to the shared Aether client."""
        try:
            agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            value.bind_client(agent_service)
        except Exception:
            logger.error("AetherTaskService failed to bind to shared Aether client", exc_info=True)

    async def async_stopping(self, v: Variables, logger: Logger, value: AetherTaskService) -> None:
        """Disconnect and cancel recurring tasks."""
        await value.disconnect()

    def get_dependencies(self, v: Variables):
        """Declare dependencies — requires storage backend and Aether agent service."""
        return (EXT_STORAGE_BACKEND, EXT_AETHER_SERVICE_CONNECTION)
