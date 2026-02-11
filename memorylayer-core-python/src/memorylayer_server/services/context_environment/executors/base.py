"""Base executor provider interface for context environment sandboxes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result from executing code in the sandbox."""

    output: str
    result: Any | None
    error: str | None
    variables_changed: list[str] = field(default_factory=list)
    operations_count: int = 0


class ExecutorProvider(ABC):
    """Pluggable sandbox execution provider."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        state: dict[str, Any],
        max_operations: int = 1_000_000,
        max_seconds: int = 30,
        max_output_chars: int = 50_000,
    ) -> ExecutionResult:
        """Execute code against persistent state dict.

        The state dict is modified in-place. The provider must enforce
        operation limits and timeouts.
        """
        ...

    @abstractmethod
    def get_allowed_modules(self) -> list[str]:
        """Return list of importable module names."""
        ...
