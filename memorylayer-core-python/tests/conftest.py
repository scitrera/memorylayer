"""
Pytest configuration and fixtures for MemoryLayer.ai tests.

Uses scitrera-app-framework dependency injection for service configuration.
Each test session gets an isolated Variables instance that does NOT pull from
environment variables - all configuration is set explicitly for test isolation.

Usage in tests:
    async def test_something(memory_service):
        result = await memory_service.remember(...)
"""
import logging
import pytest
import pytest_asyncio
import asyncio

from scitrera_app_framework import Variables, get_extension
from memorylayer_server.models.memory import RememberInput, MemoryType
from memorylayer_server.config import (
    MEMORYLAYER_EMBEDDING_PROVIDER,
    MEMORYLAYER_STORAGE_BACKEND,
    MEMORYLAYER_SQLITE_STORAGE_PATH,
    MEMORYLAYER_DATA_DIR,
    MEMORYLAYER_RERANKER_PROVIDER,
)
from memorylayer_server.services.llm.base import MEMORYLAYER_LLM_REGISTRY
from memorylayer_server.services.tasks.asyncio_impl import MEMORYLAYER_TASKS_ENABLED


# -----------------------------------------------------------------------------
# Logging Configuration (initialized by test harness, not framework)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_logger() -> logging.Logger:
    """
    Create a root logger for tests.

    The test harness owns logging configuration, not the framework.
    This prevents conflicts and ensures predictable test output.
    """
    logger = logging.getLogger("memorylayer-test")
    logger.setLevel(logging.DEBUG)

    # Add console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s %(funcName)s() > %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# -----------------------------------------------------------------------------
# Framework Initialization with Test Isolation
# -----------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="session")
async def test_configuration():
    """
    Create an isolated Variables instance to provide custom configuration
    of key environment variables for tests. The test's local framework will
    be built on top of this configuration.
    """
    # Create isolated Variables instance - set values directly
    v = Variables()
    v.set(MEMORYLAYER_EMBEDDING_PROVIDER, "mock")
    v.set(MEMORYLAYER_STORAGE_BACKEND, "sqlite")
    v.set(MEMORYLAYER_LLM_REGISTRY, "default")  # Use default registry (NoOp when no profiles configured)
    v.set(MEMORYLAYER_RERANKER_PROVIDER, "none")  # Use no-op reranker for tests
    v.set(MEMORYLAYER_TASKS_ENABLED, "false")  # Disable background tasks for tests
    return v


@pytest_asyncio.fixture(scope="session")
async def test_framework(test_configuration, tmp_path_factory, test_logger):
    """
    Initialize an isolated framework instance for the test session.

    Creates a Variables instance that does NOT pull from environment variables.
    All configuration is set explicitly for complete test isolation.

    Yields:
        tuple: (v: Variables, services: module) for use in tests
    """
    from scitrera_app_framework import Variables
    from memorylayer_server.dependencies import preconfigure, initialize_services, shutdown_services

    # Create session-scoped temp directory for database
    tmp_dir = tmp_path_factory.mktemp("memorylayer_test")

    # use overrideable test_configuration as the base for our framework
    v = test_configuration
    v.set(MEMORYLAYER_DATA_DIR, str(tmp_dir))  # set the working directory as a temp directory

    # Initialize framework in test mode (no fault handler, no pyroscope, etc.)
    v, services = preconfigure(v=v, test_mode=True, test_logger=test_logger)

    # Initialize services (connects storage, etc.)
    v = await initialize_services(v)

    yield v, services

    # Cleanup
    await shutdown_services(v)


# Convenience fixtures to unpack the tuple
@pytest.fixture(scope="session")
def v(test_framework):
    """Isolated Variables instance for tests."""
    v, _ = test_framework
    return v


#
# @pytest.fixture(scope="session")
# def services(test_framework):
#     """Services module for accessing get_*_service(v) functions."""
#     _, services = test_framework
#     return services


@pytest.fixture(scope='session')
def fastapi_app(test_framework):
    """FastAPI app instance for tests."""
    from memorylayer_server.lifecycle.fastapi import fastapi_app_factory
    v, _ = test_framework
    app = fastapi_app_factory(v=v)
    return app


# -----------------------------------------------------------------------------
# Event Loop Configuration
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# -----------------------------------------------------------------------------
# Convenience Service Fixtures
# These just call the DI system with the isolated Variables instance.
# -----------------------------------------------------------------------------

@pytest_asyncio.fixture
async def memory_service(v):
    """Get the memory service."""
    from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
    return get_extension(EXT_MEMORY_SERVICE, v)


@pytest_asyncio.fixture
async def association_service(v):
    """Get the association service."""
    from memorylayer_server.services.association import EXT_ASSOCIATION_SERVICE
    return get_extension(EXT_ASSOCIATION_SERVICE, v)


@pytest_asyncio.fixture
async def storage_backend(v):
    """Get the storage backend."""
    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
    return get_extension(EXT_STORAGE_BACKEND, v)


@pytest_asyncio.fixture
async def embedding_service(v):
    """Get the embedding service."""
    from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE
    return get_extension(EXT_EMBEDDING_SERVICE, v)


@pytest_asyncio.fixture
async def deduplication_service(v):
    """Get the deduplication service."""
    from memorylayer_server.services.deduplication import EXT_DEDUPLICATION_SERVICE
    return get_extension(EXT_DEDUPLICATION_SERVICE, v)


# -----------------------------------------------------------------------------
# Test Data Factories
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_remember_input() -> RememberInput:
    """Create sample remember input."""
    return RememberInput(
        content="User prefers Python for backend development",
        type=MemoryType.SEMANTIC,
        importance=0.8,
        tags=["preferences", "programming"],
        metadata={"source": "conversation", "confidence": 0.95}
    )


@pytest_asyncio.fixture
async def workspace_id(storage_backend) -> str:
    """Default test workspace ID with workspace and context created."""
    from memorylayer_server.models.workspace import Workspace
    from datetime import datetime, timezone

    workspace_id = "default"  # TODO: do these align with new defaults?!?!

    # Ensure workspace exists
    existing = await storage_backend.get_workspace(workspace_id)
    if not existing:
        workspace = Workspace(
            id=workspace_id,
            tenant_id="default_tenant",
            name="Default Test Workspace",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await storage_backend.create_workspace(workspace)

    return workspace_id


# -----------------------------------------------------------------------------
# Test Isolation Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def unique_workspace_id() -> str:
    """Generate a unique workspace ID for test isolation (function-scoped)."""
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="class")
def class_workspace_id(request) -> str:
    """
    Generate a unique workspace ID per test class for isolation.

    Use this for test classes that need isolated data across all their test methods.
    """
    import uuid
    class_name = request.cls.__name__ if request.cls else "unknown"
    return f"test_{class_name}_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def temp_db_path(tmp_path):
    """
    Create a temporary database path for tests that need their own storage instance.

    Use this for lifecycle tests that need to create/destroy their own storage backends.
    For normal tests, use the sqlite_storage fixture instead.
    """
    return tmp_path / "test_storage.db"
