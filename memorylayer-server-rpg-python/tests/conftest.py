# SPDX-License-Identifier: Apache-2.0
"""Pytest fixtures for memorylayer-server-rpg.

Mirrors the relevant subset of memorylayer-server's tests/conftest.py so the
moved RPG tests have the shared infrastructure they expect (an isolated
Variables instance, a session-scoped framework, a storage_backend fixture,
and a FastAPI app fixture).
"""

import asyncio
import logging

import pytest
import pytest_asyncio
from memorylayer_server.config import (
    MEMORYLAYER_DATA_DIR,
    MEMORYLAYER_EMBEDDING_PROVIDER,
    MEMORYLAYER_RERANKER_PROVIDER,
    MEMORYLAYER_STORAGE_BACKEND,
)
from memorylayer_server.services.llm.base import MEMORYLAYER_LLM_REGISTRY
from memorylayer_server.services.tasks.asyncio_impl import MEMORYLAYER_TASKS_ENABLED
from scitrera_app_framework import Variables, get_extension


@pytest.fixture(scope="session")
def test_logger() -> logging.Logger:
    logger = logging.getLogger("memorylayer-server-rpg-test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(funcName)s() > %(message)s"))
        logger.addHandler(handler)
    return logger


@pytest_asyncio.fixture(scope="session")
async def test_configuration():
    v = Variables()
    v.set(MEMORYLAYER_EMBEDDING_PROVIDER, "mock")
    v.set(MEMORYLAYER_STORAGE_BACKEND, "sqlite")
    v.set(MEMORYLAYER_LLM_REGISTRY, "default")
    v.set(MEMORYLAYER_RERANKER_PROVIDER, "none")
    v.set(MEMORYLAYER_TASKS_ENABLED, "false")
    return v


@pytest_asyncio.fixture(scope="session")
async def test_framework(test_configuration, tmp_path_factory, test_logger):
    from memorylayer_server.dependencies import initialize_services, preconfigure, shutdown_services

    tmp_dir = tmp_path_factory.mktemp("memorylayer_rpg_test")
    v = test_configuration
    v.set(MEMORYLAYER_DATA_DIR, str(tmp_dir))

    v, services = preconfigure(v=v, test_mode=True, test_logger=test_logger)
    v = await initialize_services(v)

    yield v, services

    await shutdown_services(v)


@pytest.fixture(scope="session")
def v(test_framework):
    v, _ = test_framework
    return v


@pytest.fixture(scope="session")
def fastapi_app(test_framework):
    from memorylayer_server.lifecycle.fastapi import fastapi_app_factory

    v, _ = test_framework
    return fastapi_app_factory(v=v)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def storage_backend(v):
    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

    return get_extension(EXT_STORAGE_BACKEND, v)
