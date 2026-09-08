"""Regression tests for TaskHandlersSetupPlugin's handler/schedule loops.

The EXT_MULTI_TASK_HANDLERS multi-extension collection carries not only the
TaskHandlerPlugin instances but also TaskHandlersSetupPlugin's own initialize()
return value -- the TaskService, which is NOT a handler and has no
get_schedule()/get_task_type(). Both loops must skip such non-handler entries so
a single poison entry can't abort registration/scheduling for every handler
ordered after it (which manifested as the startup warning:
``'AetherTaskService' object has no attribute 'get_schedule'`` leaving later
recurring sweeps unscheduled).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_server.services.tasks import handlers as handlers_mod
from memorylayer_server.services.tasks.handlers import (
    TaskHandlerPlugin,
    TaskHandlersSetupPlugin,
)
from memorylayer_server.services.tasks.base import TaskSchedule


class _RecurringHandler(TaskHandlerPlugin):
    """Minimal recurring handler used to populate the registry."""

    def __init__(self, task_type: str, interval: int = 60) -> None:
        self._task_type = task_type
        self._interval = interval

    def get_task_type(self) -> str:
        return self._task_type

    async def handle(self, v, payload) -> None:  # pragma: no cover - not invoked
        return None

    def get_schedule(self, v):
        return TaskSchedule(interval_seconds=self._interval, default_payload={})


class _NotAHandler:
    """Stand-in for the TaskService object that pollutes the collection.

    Deliberately lacks get_schedule() / get_task_type() so touching it raises
    AttributeError -- exactly the AetherTaskService case from production.
    """


def _registry_with_poison_in_the_middle():
    """Ordered registry: a real handler, then the poison entry, then another
    handler -- so a non-skipping loop would schedule only the first."""
    return {
        "first": _RecurringHandler("first_sweep"),
        "poison": _NotAHandler(),  # e.g. the AetherTaskService return value
        "second": _RecurringHandler("second_sweep"),
    }


@pytest.mark.asyncio
async def test_async_ready_skips_non_handler_and_schedules_all_handlers():
    task_service = MagicMock()
    task_service.schedule_recurring = AsyncMock()

    registry = _registry_with_poison_in_the_middle()
    with patch.object(handlers_mod, "get_extensions", return_value=registry):
        await TaskHandlersSetupPlugin().async_ready(MagicMock(), MagicMock(), task_service)

    scheduled = {c.args[0] for c in task_service.schedule_recurring.await_args_list}
    # The handler ordered AFTER the poison entry must still be scheduled.
    assert scheduled == {"first_sweep", "second_sweep"}, scheduled


def test_initialize_skips_non_handler_and_registers_all_handlers():
    task_service = MagicMock()
    registry = _registry_with_poison_in_the_middle()

    plugin = TaskHandlersSetupPlugin()
    with patch.object(handlers_mod, "get_extensions", return_value=registry), \
            patch.object(plugin, "get_extension", return_value=task_service), \
            patch.object(handlers_mod, "ext_parse_bool", side_effect=lambda x: x):
        v = MagicMock()
        v.environ.return_value = True  # in-process worker enabled
        result = plugin.initialize(v, MagicMock())

    registered = {c.args[0] for c in task_service.register_handler.call_args_list}
    assert registered == {"first_sweep", "second_sweep"}, registered
    assert result is task_service
