"""Default context environment service implementation.

Provides in-memory sandboxed Python environments per session with
memory integration, LLM queries, and iterative reasoning loops.
"""
import json
import sys
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Optional

from scitrera_app_framework import get_logger, get_extension, Variables

from .base import (
    ContextEnvironmentService,
    ContextEnvironmentServicePluginBase,
    EXT_CONTEXT_ENVIRONMENT_SERVICE,
)
from .executors.base import ExecutorProvider, ExecutionResult
from .hooks import ContextPersistenceHook, NoOpPersistenceHook
from ...config import (
    MEMORYLAYER_CONTEXT_EXECUTOR,
    DEFAULT_MEMORYLAYER_CONTEXT_EXECUTOR,
    MEMORYLAYER_CONTEXT_MAX_OPERATIONS,
    DEFAULT_MEMORYLAYER_CONTEXT_MAX_OPERATIONS,
    MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS,
    DEFAULT_MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS,
    MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS,
    DEFAULT_MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS,
    MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS,
    MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES,
    DEFAULT_MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES,
    MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
    DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
    MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
    DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
)


def _safe_preview(value: Any, max_chars: int = 200) -> str:
    """Generate a safe string preview of a value."""
    try:
        s = repr(value)
    except Exception:
        s = f"<{type(value).__name__}>"
    if len(s) > max_chars:
        return s[:max_chars] + '...'
    return s


def _estimate_size(value: Any) -> int:
    """Estimate memory size of a value in bytes."""
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def _memory_to_dict(memory: Any, include_embeddings: bool = False) -> dict:
    """Convert a Memory model to a plain dict for sandbox use."""
    d = {
        'id': memory.id,
        'content': memory.content,
        'type': str(memory.type.value) if memory.type else None,
        'importance': memory.importance,
        'tags': list(memory.tags) if memory.tags else [],
        'created_at': memory.created_at.isoformat() if memory.created_at else None,
    }
    if memory.metadata:
        d['metadata'] = dict(memory.metadata)
    if memory.abstract:
        d['abstract'] = memory.abstract
    if include_embeddings and memory.embedding:
        d['embedding'] = list(memory.embedding)
    return d


class DefaultContextEnvironmentService(ContextEnvironmentService):
    """Default in-memory implementation of the context environment service.

    Each session gets its own sandbox state dict managed by an executor provider.
    State persists across calls within a session but is lost on service restart.
    """

    def __init__(
        self,
        v: Variables,
        executor: ExecutorProvider,
        persistence_hook: ContextPersistenceHook | None = None,
    ):
        """Initialize the default context environment service.

        Args:
            v: Application variables for config and service access
            executor: The sandbox execution provider
            persistence_hook: Optional hook for state persistence
        """
        self._v = v
        self._executor = executor
        self._hook = persistence_hook or NoOpPersistenceHook()
        self.logger = get_logger(v, name=self.__class__.__name__)

        # Per-session sandbox state
        self._environments: dict[str, dict[str, Any]] = {}
        # Per-session metadata tracking
        self._env_metadata: dict[str, dict[str, Any]] = {}

        # Load config
        self._max_operations = int(v.get(
            MEMORYLAYER_CONTEXT_MAX_OPERATIONS,
            DEFAULT_MEMORYLAYER_CONTEXT_MAX_OPERATIONS,
        ))
        self._max_exec_seconds = int(v.get(
            MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS,
            DEFAULT_MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS,
        ))
        self._max_output_chars = int(v.get(
            MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS,
            DEFAULT_MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS,
        ))
        self._query_max_tokens = int(v.get(
            MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS,
            DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS,
        ))
        self._max_memory_bytes = int(v.get(
            MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES,
            DEFAULT_MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES,
        ))
        self._exec_soft_cap = int(v.get(
            MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
            DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
        ))
        self._exec_hard_cap = int(v.get(
            MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
            DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
        ))

        self.logger.info("DefaultContextEnvironmentService initialized")

    async def _init_environment(self, session_id: str) -> dict[str, Any]:
        """Get or create and potentially restore the sandbox state for a session."""
        if session_id in self._environments:
            return self._environments[session_id]

        # Try to restore from persistence hook
        restored_state = await self._hook.on_session_restore(session_id)
        if restored_state is not None:
            self._environments[session_id] = restored_state
            self._env_metadata[session_id] = {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'exec_count': 0,
                'total_operations': 0,
                'restored': True,
            }
            self.logger.info("Restored environment for session %s from persistence hook", session_id)
        else:
            self._environments[session_id] = {}
            self._env_metadata[session_id] = {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'exec_count': 0,
                'total_operations': 0,
            }
            self.logger.info("Created environment for session: %s", session_id)

        return self._environments[session_id]

    def _check_rate_limits(self, session_id: str) -> str | None:
        """Check rate limits. Returns error message if exceeded, None if ok."""
        meta = self._env_metadata.get(session_id, {})
        exec_count = meta.get('exec_count', 0)

        if self._exec_hard_cap > 0 and exec_count >= self._exec_hard_cap:
            return f"Hard execution cap reached: {exec_count} >= {self._exec_hard_cap}"

        if self._exec_soft_cap > 0 and exec_count >= self._exec_soft_cap:
            self.logger.warning(
                "Soft execution cap reached for session %s: %d >= %d",
                session_id, exec_count, self._exec_soft_cap,
            )

        return None

    def _check_memory_limit(self, session_id: str) -> str | None:
        """Check memory usage limit. Returns error message if exceeded."""
        if self._max_memory_bytes <= 0:
            return None

        state = self._environments.get(session_id, {})
        total_size = sum(_estimate_size(v) for v in state.values())

        if total_size > self._max_memory_bytes:
            return (
                f"Memory limit exceeded: {total_size} bytes > "
                f"{self._max_memory_bytes} byte limit"
            )
        return None

    async def execute(
        self,
        session_id: str,
        code: str,
        result_var: Optional[str] = None,
        return_result: bool = True,
        max_return_chars: int = 10_000,
    ) -> dict:
        """Execute code in the session's sandbox environment."""
        # Rate limit check
        rate_error = self._check_rate_limits(session_id)
        if rate_error:
            return {'output': '', 'result': None, 'error': rate_error, 'variables_changed': []}

        state = await self._init_environment(session_id)

        # Memory limit check
        mem_error = self._check_memory_limit(session_id)
        if mem_error:
            return {'output': '', 'result': None, 'error': mem_error, 'variables_changed': []}

        self.logger.debug("Executing code in session %s: %s", session_id, code[:100])

        result: ExecutionResult = await self._executor.execute(
            code=code,
            state=state,
            max_operations=self._max_operations,
            max_seconds=self._max_exec_seconds,
            max_output_chars=self._max_output_chars,
        )

        # Update metadata
        meta = self._env_metadata[session_id]
        meta['exec_count'] = meta.get('exec_count', 0) + 1
        meta['total_operations'] = meta.get('total_operations', 0) + result.operations_count
        meta['last_exec_at'] = datetime.now(timezone.utc).isoformat()

        # Store result in variable if requested
        if result_var and result.result is not None and result.error is None:
            state[result_var] = result.result
            if result_var not in result.variables_changed:
                result.variables_changed.append(result_var)

        # Notify persistence hook
        if result.variables_changed and result.error is None:
            await self._hook.on_state_changed(session_id, state)

        # Build response
        response: dict[str, Any] = {
            'output': result.output,
            'error': result.error,
            'variables_changed': result.variables_changed,
        }

        if return_result and result.result is not None:
            preview = _safe_preview(result.result, max_return_chars)
            response['result'] = preview
        else:
            response['result'] = None

        return response

    async def inspect(
        self,
        session_id: str,
        variable: Optional[str] = None,
        preview_chars: int = 200,
    ) -> dict:
        """Inspect sandbox state or a specific variable."""
        state = await self._init_environment(session_id)

        if variable is not None:
            if variable not in state:
                return {'error': f"Variable '{variable}' not found"}
            value = state[variable]
            return {
                'variable': variable,
                'type': type(value).__name__,
                'preview': _safe_preview(value, preview_chars),
                'size_bytes': _estimate_size(value),
            }

        # Return overview of all variables
        variables = {}
        for key, value in state.items():
            variables[key] = {
                'type': type(value).__name__,
                'preview': _safe_preview(value, preview_chars),
                'size_bytes': _estimate_size(value),
            }

        return {
            'variable_count': len(variables),
            'variables': variables,
            'total_size_bytes': sum(v['size_bytes'] for v in variables.values()),
        }

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
        """Load memories into the sandbox as a variable."""
        state = await self._init_environment(session_id)

        # Rate limit check
        rate_error = self._check_rate_limits(session_id)
        if rate_error:
            return {'error': rate_error, 'count': 0}

        try:
            from ..memory import get_memory_service
            from ..session import get_session_service
            from ...models.memory import RecallInput, MemoryType

            # Resolve the session to get workspace_id
            session_service = get_session_service(self._v)
            session = await session_service.get(session_id)
            if session is None:
                return {'error': f"Session not found: {session_id}", 'count': 0}

            # Build recall input
            type_filters = []
            if types:
                for t in types:
                    try:
                        type_filters.append(MemoryType(t))
                    except ValueError:
                        self.logger.warning("Unknown memory type filter: %s", t)

            recall_input = RecallInput(
                query=query,
                limit=limit,
                types=type_filters,
                tags=tags or [],
                min_relevance=min_relevance,
            )

            memory_service = get_memory_service(self._v)
            recall_result = await memory_service.recall(
                workspace_id=session.workspace_id,
                input=recall_input,
            )

            # Convert memories to dicts and store in sandbox
            memory_dicts = [
                _memory_to_dict(m, include_embeddings=include_embeddings)
                for m in recall_result.memories
            ]
            state[var] = memory_dicts

            # Notify persistence hook
            await self._hook.on_state_changed(session_id, state)

            self.logger.info(
                "Loaded %d memories into session %s variable '%s'",
                len(memory_dicts), session_id, var,
            )

            return {
                'count': len(memory_dicts),
                'variable': var,
                'query': query,
                'total_available': recall_result.total_count,
            }

        except ImportError as e:
            return {'error': f"Memory service not available: {e}", 'count': 0}
        except Exception as e:
            self.logger.error("Failed to load memories for session %s: %s", session_id, e, exc_info=True)
            return {'error': f"Memory load failed: {e}", 'count': 0}

    async def inject(
        self,
        session_id: str,
        key: str,
        value: Any,
        parse_json: bool = False,
    ) -> dict:
        """Inject a value into the sandbox state."""
        state = await self._init_environment(session_id)

        # Rate limit check
        rate_error = self._check_rate_limits(session_id)
        if rate_error:
            return {'error': rate_error}

        if parse_json and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                return {'error': f"JSON parse error: {e}"}

        state[key] = value

        # Notify persistence hook
        await self._hook.on_state_changed(session_id, state)

        self.logger.debug("Injected variable '%s' into session %s", key, session_id)

        return {
            'variable': key,
            'type': type(value).__name__,
            'preview': _safe_preview(value, 200),
        }

    async def query(
        self,
        session_id: str,
        prompt: str,
        variables: list[str],
        max_context_chars: Optional[int] = None,
        result_var: Optional[str] = None,
    ) -> dict:
        """Send sandbox variables and a prompt to the LLM."""
        state = await self._init_environment(session_id)
        max_chars = max_context_chars or self._max_output_chars

        try:
            from ..llm import get_llm_service

            # Build context from specified variables
            context_parts = []
            for var_name in variables:
                if var_name not in state:
                    context_parts.append(f"[{var_name}]: (not found)")
                    continue
                value = state[var_name]
                preview = _safe_preview(value, max_chars // max(len(variables), 1))
                context_parts.append(f"[{var_name}] ({type(value).__name__}):\n{preview}")

            context = '\n\n'.join(context_parts)

            llm_service = get_llm_service(self._v)
            response_text = await llm_service.synthesize(
                prompt=prompt,
                context=context,
                max_tokens=self._query_max_tokens,
            )

            # Store result if requested
            if result_var:
                state[result_var] = response_text
                await self._hook.on_state_changed(session_id, state)

            self.logger.info("LLM query completed for session %s", session_id)

            return {
                'response': response_text,
                'variables_used': variables,
                'result_var': result_var,
            }

        except ImportError as e:
            return {'error': f"LLM service not available: {e}"}
        except Exception as e:
            self.logger.error("LLM query failed for session %s: %s", session_id, e, exc_info=True)
            return {'error': f"LLM query failed: {e}"}

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

        Delegates to the RLM runner for iterative execution.
        """
        from .rlm import RLMRunner

        runner = RLMRunner(
            service=self,
            v=self._v,
        )

        return await runner.run(
            session_id=session_id,
            goal=goal,
            memory_query=memory_query,
            memory_limit=memory_limit,
            max_iterations=max_iterations,
            variables=variables or [],
            result_var=result_var,
            detail_level=detail_level,
        )

    async def status(self, session_id: str) -> dict:
        """Get the status of a session's sandbox environment."""
        if session_id not in self._environments:
            return {
                'exists': False,
                'variable_count': 0,
                'total_size_bytes': 0,
                'metadata': {},
            }

        state = self._environments[session_id]
        meta = self._env_metadata.get(session_id, {})

        total_size = sum(_estimate_size(v) for v in state.values())

        return {
            'exists': True,
            'variable_count': len(state),
            'variables': list(state.keys()),
            'total_size_bytes': total_size,
            'memory_limit_bytes': self._max_memory_bytes,
            'metadata': meta,
        }

    async def cleanup_environment(self, session_id: str) -> None:
        """Clean up and remove a session's sandbox environment."""
        if session_id in self._environments:
            state = self._environments[session_id]

            # Notify persistence hook before cleanup
            await self._hook.on_session_end(session_id, state)

            del self._environments[session_id]
            self.logger.info("Cleaned up environment for session: %s", session_id)

        if session_id in self._env_metadata:
            del self._env_metadata[session_id]

    async def checkpoint(self, session_id: str) -> None:
        """Checkpoint the session's sandbox state for persistence."""
        if session_id in self._environments:
            state = self._environments[session_id]
            await self._hook.on_checkpoint(session_id, state)
            self.logger.info("Checkpoint fired for session: %s", session_id)


class DefaultContextEnvironmentServicePlugin(ContextEnvironmentServicePluginBase):
    """Plugin for the default context environment service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> ContextEnvironmentService:
        """Initialize the default context environment service."""
        executor_type = v.get(
            MEMORYLAYER_CONTEXT_EXECUTOR,
            DEFAULT_MEMORYLAYER_CONTEXT_EXECUTOR,
        )

        executor: ExecutorProvider
        if executor_type == 'smolagents':
            try:
                from .executors.smolagents_executor import SmolagentsExecutor
                executor = SmolagentsExecutor()
                logger.info("Using smolagents executor for context environments")
            except ImportError:
                logger.warning(
                    "smolagents not available, falling back to restricted executor"
                )
                from .executors.restricted import RestrictedExecutor
                executor = RestrictedExecutor()
        elif executor_type == 'restricted':
            from .executors.restricted import RestrictedExecutor
            executor = RestrictedExecutor()
            logger.info("Using restricted executor for context environments")
        else:
            logger.warning(
                "Unknown executor type '%s', falling back to restricted", executor_type
            )
            from .executors.restricted import RestrictedExecutor
            executor = RestrictedExecutor()

        return DefaultContextEnvironmentService(
            v=v,
            executor=executor,
        )
