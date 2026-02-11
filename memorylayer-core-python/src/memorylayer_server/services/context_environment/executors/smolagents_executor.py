"""Smolagents-based executor provider for context environment sandboxes.

Wraps the smolagents LocalPythonExecutor to provide a sandboxed Python
execution environment with controlled imports and built-in functions.
"""
import asyncio
import logging
from typing import Any

from .base import ExecutorProvider, ExecutionResult

logger = logging.getLogger(__name__)

# Modules allowed for import in the sandbox
_IMPORT_WHITELIST = [
    'collections',
    'datetime',
    'itertools',
    'math',
    'queue',
    'random',
    're',
    'stat',
    'statistics',
    'time',
    'unicodedata',
    'json',
    'functools',
]


class SmolagentsExecutor(ExecutorProvider):
    """Executor provider wrapping smolagents LocalPythonExecutor.

    Provides a full Python sandbox with:
    - Safe built-in functions (sum, len, sorted, etc.)
    - Controlled module imports
    - State persistence across executions
    - Timeout enforcement
    - Custom injected functions (llm_query, SUBMIT, etc.)
    """

    def __init__(
        self,
        additional_imports: list[str] | None = None,
        additional_functions: dict[str, Any] | None = None,
        max_print_output_length: int | None = None,
    ):
        """Initialize the smolagents executor.

        Args:
            additional_imports: Extra module names to allow beyond the default whitelist
            additional_functions: Extra callable functions to inject into the sandbox
            max_print_output_length: Maximum length for captured print output
        """
        self._additional_imports = additional_imports or []
        self._additional_functions = additional_functions or {}
        self._max_print_output_length = max_print_output_length
        self._executor = None

    def _ensure_executor(self) -> None:
        """Lazily create the smolagents executor on first use."""
        if self._executor is not None:
            return

        try:
            from smolagents.local_python_executor import LocalPythonExecutor
        except ImportError as exc:
            raise RuntimeError(
                "smolagents package is required for SmolagentsExecutor. "
                "Install it with: pip install 'smolagents>=1.0,<2.0'"
            ) from exc

        all_imports = list(set(_IMPORT_WHITELIST) | set(self._additional_imports))

        self._executor = LocalPythonExecutor(
            additional_authorized_imports=all_imports,
            max_print_outputs_length=self._max_print_output_length,
            additional_functions=self._additional_functions,
        )
        # Initialize static_tools so BASE_PYTHON_TOOLS are available
        self._executor.send_tools({})

        logger.debug(
            "SmolagentsExecutor initialized with %d authorized imports",
            len(all_imports),
        )

    async def execute(
        self,
        code: str,
        state: dict[str, Any],
        max_operations: int = 1_000_000,
        max_seconds: int = 30,
        max_output_chars: int = 50_000,
    ) -> ExecutionResult:
        """Execute code using the smolagents sandbox.

        Args:
            code: Python code to execute
            state: Persistent state dict, synced bidirectionally with the sandbox
            max_operations: Not directly enforced by smolagents (tracked for reporting)
            max_seconds: Maximum execution time in seconds
            max_output_chars: Maximum characters in captured output

        Returns:
            ExecutionResult with output, result, and change tracking
        """
        code = code.strip()
        if not code:
            return ExecutionResult(output='', result=None, error=None)

        self._ensure_executor()

        # Sync external state into the smolagents sandbox
        keys_before = set(state.keys())
        values_before = {k: id(v) for k, v in state.items()}

        # Inject current state into the executor
        self._executor.send_variables(state)

        # Override timeout if different from default
        original_timeout = self._executor.timeout_seconds
        if max_seconds != original_timeout:
            self._executor.timeout_seconds = max_seconds

        try:
            from smolagents.local_python_executor import InterpreterError

            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            try:
                code_output = await loop.run_in_executor(
                    None, self._executor, code
                )
            except InterpreterError as e:
                return ExecutionResult(
                    output='',
                    result=None,
                    error=f"InterpreterError: {e}",
                )
            except TimeoutError:
                return ExecutionResult(
                    output='',
                    result=None,
                    error=f"Execution timed out after {max_seconds}s",
                )
            except Exception as e:
                return ExecutionResult(
                    output='',
                    result=None,
                    error=f"{type(e).__name__}: {e}",
                )

            # Extract results
            output = code_output.logs or ''
            if len(output) > max_output_chars:
                output = output[:max_output_chars]

            result_value = code_output.output

            # Track operations count from smolagents internal counter
            ops_counter = self._executor.state.get('_operations_count', {})
            operations_count = ops_counter.get('counter', 0) if isinstance(ops_counter, dict) else 0

            # Sync state changes back
            variables_changed: list[str] = []
            executor_state = self._executor.state

            for key in list(executor_state.keys()):
                if key.startswith('_'):
                    continue
                if key not in keys_before:
                    # New variable
                    state[key] = executor_state[key]
                    variables_changed.append(key)
                elif id(executor_state[key]) != values_before.get(key):
                    # Changed variable (identity check)
                    state[key] = executor_state[key]
                    variables_changed.append(key)
                else:
                    # Ensure value is synced even if identity matches
                    state[key] = executor_state[key]

            # Track deletions
            executor_user_keys = {k for k in executor_state if not k.startswith('_')}
            for deleted_key in keys_before - executor_user_keys:
                if deleted_key in state:
                    del state[deleted_key]
                    variables_changed.append(deleted_key)

            return ExecutionResult(
                output=output,
                result=result_value,
                error=None,
                variables_changed=variables_changed,
                operations_count=operations_count,
            )

        finally:
            # Restore original timeout
            if max_seconds != original_timeout:
                self._executor.timeout_seconds = original_timeout

    def get_allowed_modules(self) -> list[str]:
        """Return list of importable module names."""
        return list(set(_IMPORT_WHITELIST) | set(self._additional_imports))
