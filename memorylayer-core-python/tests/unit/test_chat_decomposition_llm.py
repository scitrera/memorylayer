"""Regression test: chat decomposition must call LLMService.complete() (not the
non-existent .generate()) and extract memories from the response."""

import logging
from datetime import UTC, datetime

import pytest
from scitrera_app_framework.api import Variables

from memorylayer_server.models.chat import ChatMessage, ChatThread
from memorylayer_server.models.llm import LLMResponse
from memorylayer_server.tasks.chat_decomposition_handler import ChatDecompositionTaskHandler


class _FakeLLMService:
    """Only exposes complete() — NOT generate(); proves the handler uses the
    real LLMService API."""

    def __init__(self, content: str):
        self._content = content
        self.calls = []

    async def complete(self, request, profile: str = "default", **_generation_metadata):
        self.calls.append((request, profile))
        return LLMResponse(
            content=self._content, model="fake", prompt_tokens=1,
            completion_tokens=1, total_tokens=2, finish_reason="stop",
        )


class _FakeMemoryService:
    def __init__(self):
        self.remembered = []

    async def remember(self, workspace_id, remember_input):
        self.remembered.append((workspace_id, remember_input))
        return None


def _thread():
    now = datetime.now(UTC)
    return ChatThread(id="t1", workspace_id="ws", tenant_id="_default",
                      created_at=now, updated_at=now)


def _msgs(n=3):
    return [
        ChatMessage(id=f"m{i}", thread_id="t1", message_index=i,
                    role="user" if i % 2 == 0 else "assistant", content=f"line {i}")
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_decompose_chunk_uses_complete_and_creates_memories():
    handler = ChatDecompositionTaskHandler()
    llm = _FakeLLMService(
        '[{"content": "user prefers dark mode", "type": "semantic", "importance": 0.7}]'
    )
    # Inject the fake LLM service for the EXT_LLM_SERVICE lookup.
    handler.get_extension = lambda ext, v: llm
    mem = _FakeMemoryService()

    created = await handler._decompose_chunk(
        v=Variables(),
        logger=logging.getLogger("test-decompose"),
        storage=None,
        memory_service=mem,
        workspace_id="ws",
        thread=_thread(),
        messages=_msgs(),
    )

    assert created == 1
    assert len(mem.remembered) == 1
    # complete() was called with a system + user message request.
    assert llm.calls, "LLMService.complete was not called"
    request, _profile = llm.calls[0]
    role_values = [getattr(m.role, "value", str(m.role)) for m in request.messages]
    assert "system" in role_values and "user" in role_values


@pytest.mark.asyncio
async def test_decompose_chunk_handles_empty_extraction():
    handler = ChatDecompositionTaskHandler()
    handler.get_extension = lambda ext, v: _FakeLLMService("[]")
    created = await handler._decompose_chunk(
        v=Variables(), logger=logging.getLogger("t"), storage=None,
        memory_service=_FakeMemoryService(), workspace_id="ws",
        thread=_thread(), messages=_msgs(),
    )
    assert created == 0


def _msgs_with_system():
    """A chunk mixing a system message with real user/assistant turns."""
    return [
        ChatMessage(id="m0", thread_id="t1", message_index=0,
                    role="system", content="SECRET_SYSTEM_PROMPT"),
        ChatMessage(id="m1", thread_id="t1", message_index=1,
                    role="user", content="USER_LINE"),
        ChatMessage(id="m2", thread_id="t1", message_index=2,
                    role="assistant", content="ASSISTANT_LINE"),
    ]


@pytest.mark.asyncio
async def test_decompose_chunk_excludes_system_messages():
    """System-role messages must not reach the extraction LLM."""
    handler = ChatDecompositionTaskHandler()
    llm = _FakeLLMService(
        '[{"content": "user prefers dark mode", "type": "semantic", "importance": 0.7}]'
    )
    handler.get_extension = lambda ext, v: llm

    created = await handler._decompose_chunk(
        v=Variables(), logger=logging.getLogger("t"), storage=None,
        memory_service=_FakeMemoryService(), workspace_id="ws",
        thread=_thread(), messages=_msgs_with_system(),
    )

    assert created == 1
    assert llm.calls, "LLMService.complete was not called"
    request, _profile = llm.calls[0]
    conversation = " ".join(str(m.content) for m in request.messages)
    assert "USER_LINE" in conversation
    assert "ASSISTANT_LINE" in conversation
    # The system message's content is excluded from the decomposition excerpt.
    assert "SECRET_SYSTEM_PROMPT" not in conversation


@pytest.mark.asyncio
async def test_decompose_chunk_all_system_skips_llm():
    """An all-system chunk yields no facts and never calls the LLM."""
    handler = ChatDecompositionTaskHandler()
    llm = _FakeLLMService("[]")
    handler.get_extension = lambda ext, v: llm
    sys_msgs = [
        ChatMessage(id=f"s{i}", thread_id="t1", message_index=i,
                    role="system", content=f"sys {i}")
        for i in range(3)
    ]

    created = await handler._decompose_chunk(
        v=Variables(), logger=logging.getLogger("t"), storage=None,
        memory_service=_FakeMemoryService(), workspace_id="ws",
        thread=_thread(), messages=sys_msgs,
    )

    assert created == 0
    assert llm.calls == []  # short-circuits before any LLM call
