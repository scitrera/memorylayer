"""Context environment service - ABC and plugin base.

The context environment service provides sandboxed Python execution
environments tied to sessions, with access to memory recall and LLM queries.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import (
    MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
    DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
)

from .._constants import EXT_CONTEXT_ENVIRONMENT_SERVICE


class ContextEnvironmentService(ABC):
    """Interface for context environment service.

    Provides sandboxed Python execution environments per session with
    memory integration, LLM queries, and iterative reasoning loops.
    """

    @abstractmethod
    async def execute(
        self,
        session_id: str,
        code: str,
        result_var: Optional[str] = None,
        return_result: bool = True,
        max_return_chars: int = 10_000,
    ) -> dict:
        """Execute code in the session's sandbox environment.

        Args:
            session_id: Session identifier
            code: Python code to execute
            result_var: If set, store the expression result in this variable
            return_result: Whether to include the result value in response
            max_return_chars: Maximum characters for result serialization

        Returns:
            Dict with keys: output, result, error, variables_changed
        """
        ...

    @abstractmethod
    async def inspect(
        self,
        session_id: str,
        variable: Optional[str] = None,
        preview_chars: int = 200,
    ) -> dict:
        """Inspect sandbox state or a specific variable.

        Args:
            session_id: Session identifier
            variable: Specific variable name to inspect (all if None)
            preview_chars: Maximum characters per variable preview

        Returns:
            Dict with variable names, types, and preview values
        """
        ...

    @abstractmethod
    async def load(
        self,
        session_id: str,
        var: str,
        query: str,
        limit: int = 50,
        types: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        min_relevance: Optional[float] = None,
        include_embeddings: bool = False,
    ) -> dict:
        """Load memories into the sandbox as a variable.

        Runs a memory recall query and stores the results as a list
        of dicts in the specified variable.

        Args:
            session_id: Session identifier
            var: Variable name to store results in
            query: Memory recall query
            limit: Maximum memories to recall
            types: Filter by memory types
            tags: Filter by tags
            min_relevance: Minimum relevance score
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dict with count and variable info
        """
        ...

    @abstractmethod
    async def inject(
        self,
        session_id: str,
        key: str,
        value: Any,
        parse_json: bool = False,
    ) -> dict:
        """Inject a value into the sandbox state.

        Args:
            session_id: Session identifier
            key: Variable name
            value: Value to inject
            parse_json: If True, parse value as JSON string

        Returns:
            Dict confirming the injection
        """
        ...

    @abstractmethod
    async def query(
        self,
        session_id: str,
        prompt: str,
        variables: list[str],
        max_context_chars: Optional[int] = None,
        result_var: Optional[str] = None,
    ) -> dict:
        """Send sandbox variables and a prompt to the LLM.

        Args:
            session_id: Session identifier
            prompt: User prompt for the LLM
            variables: Variable names to include as context
            max_context_chars: Maximum characters for variable context
            result_var: If set, store the LLM response in this variable

        Returns:
            Dict with the LLM response text and token usage
        """
        ...

    @abstractmethod
    async def rlm(
        self,
        session_id: str,
        goal: str,
        memory_query: Optional[str] = None,
        memory_limit: int = 100,
        max_iterations: int = 10,
        variables: Optional[list[str]] = None,
        result_var: Optional[str] = None,
        detail_level: str = "standard",
    ) -> dict:
        """Run a Recursive Language Model (RLM) loop.

        Iteratively executes code and LLM queries to achieve a goal.

        Args:
            session_id: Session identifier
            goal: Natural language description of the goal
            memory_query: Optional memory query to load initial data
            memory_limit: Maximum memories to load
            max_iterations: Maximum reasoning iterations
            variables: Variable names to include in context
            result_var: If set, store the final result in this variable
            detail_level: Level of detail: "minimal", "standard", "verbose"

        Returns:
            Dict with result, iterations performed, and execution trace
        """
        ...

    @abstractmethod
    async def status(self, session_id: str) -> dict:
        """Get the status of a session's sandbox environment.

        Args:
            session_id: Session identifier

        Returns:
            Dict with variable count, memory usage, and metadata
        """
        ...

    @abstractmethod
    async def cleanup_environment(self, session_id: str) -> None:
        """Clean up and remove a session's sandbox environment.

        Args:
            session_id: Session identifier
        """
        ...

    @abstractmethod
    async def checkpoint(self, session_id: str) -> None:
        """Checkpoint the session's sandbox state for persistence.

        Fires the on_checkpoint persistence hook, allowing enterprise
        implementations to persist sandbox state to durable storage.

        Args:
            session_id: Session identifier
        """
        ...


# noinspection PyAbstractClass
class ContextEnvironmentServicePluginBase(Plugin):
    """Base plugin for context environment service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CONTEXT_ENVIRONMENT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CONTEXT_ENVIRONMENT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(
            self, v, MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE, self_attr='PROVIDER_NAME'
        )

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
            DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
        )
