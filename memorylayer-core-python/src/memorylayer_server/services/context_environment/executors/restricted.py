"""Restricted expression-only executor using Python ast module.

Supports a safe subset of Python via AST node whitelisting:
- Variable assignment, attribute access, subscript/slice
- List/dict comprehensions
- Built-in functions (len, sorted, sum, min, max, etc.)
- String methods
- Comparisons, boolean logic, arithmetic
- del statements
- Multiline code (sequential statements)

Does NOT support:
- import, for/while loops, def/class, exec/eval
- File I/O, any module access
"""
import ast
import io
import sys
import time
from contextlib import redirect_stdout
from typing import Any

from .base import ExecutorProvider, ExecutionResult


# Safe built-in functions available in the sandbox
_SAFE_BUILTINS = {
    'len': len,
    'sorted': sorted,
    'sum': sum,
    'min': min,
    'max': max,
    'filter': filter,
    'map': map,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'abs': abs,
    'round': round,
    'enumerate': enumerate,
    'zip': zip,
    'range': range,
    'type': type,
    'isinstance': isinstance,
    'any': any,
    'all': all,
    'reversed': reversed,
    'hash': hash,
    'repr': repr,
    'print': print,
    'None': None,
    'True': True,
    'False': False,
}

# AST node types that are allowed
_ALLOWED_EXPR_NODES = (
    # Literals
    ast.Constant,
    ast.FormattedValue,
    ast.JoinedStr,
    # Collections
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    # Comprehensions
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    # Operations
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    # Accessors
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Starred,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.MatMult,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Invert,
    ast.UAdd,
    ast.USub,
    # Comparisons
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    # Call
    ast.Call,
    ast.keyword,
)

_ALLOWED_STMT_NODES = (
    ast.Expr,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Delete,
    ast.If,
)

_ALLOWED_TOP_NODES = (
    ast.Module,
    ast.Interactive,
    ast.Expression,
)


class _ASTValidator(ast.NodeVisitor):
    """Validates that an AST only contains allowed node types."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def _check_node(self, node: ast.AST) -> None:
        """Check if a node type is allowed."""
        if isinstance(node, _ALLOWED_TOP_NODES):
            return
        if isinstance(node, _ALLOWED_stmt_nodes_tuple):
            return
        if isinstance(node, _ALLOWED_EXPR_NODES):
            return
        self.errors.append(
            f"Disallowed syntax: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
        )

    def generic_visit(self, node: ast.AST) -> None:
        self._check_node(node)
        super().generic_visit(node)


# Combined tuple for statement checking
_ALLOWED_stmt_nodes_tuple = _ALLOWED_STMT_NODES


def _validate_ast(tree: ast.AST) -> list[str]:
    """Validate an AST tree and return list of errors."""
    validator = _ASTValidator()
    validator.visit(tree)
    return validator.errors


class RestrictedExecutor(ExecutorProvider):
    """Expression-only executor using Python ast module with node whitelisting."""

    async def execute(
        self,
        code: str,
        state: dict[str, Any],
        max_operations: int = 1_000_000,
        max_seconds: int = 30,
        max_output_chars: int = 50_000,
    ) -> ExecutionResult:
        """Execute restricted Python code against the state dict.

        Args:
            code: Python code to execute (restricted subset)
            state: Persistent state dict, modified in-place
            max_operations: Maximum AST nodes allowed (complexity limit)
            max_seconds: Maximum wall-clock seconds
            max_output_chars: Maximum characters in captured output

        Returns:
            ExecutionResult with output, result, and change tracking
        """
        code = code.strip()
        if not code:
            return ExecutionResult(output='', result=None, error=None)

        # Parse and validate AST
        try:
            tree = ast.parse(code, mode='exec')
        except SyntaxError as e:
            return ExecutionResult(
                output='',
                result=None,
                error=f"Syntax error: {e}",
            )

        # Count AST nodes as a complexity proxy
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > max_operations:
            return ExecutionResult(
                output='',
                result=None,
                error=f"Code complexity exceeds limit: {node_count} nodes > {max_operations} max",
                operations_count=node_count,
            )

        # Validate AST nodes
        errors = _validate_ast(tree)
        if errors:
            return ExecutionResult(
                output='',
                result=None,
                error='; '.join(errors),
                operations_count=node_count,
            )

        # Build execution namespace with safe builtins and current state
        namespace = {'__builtins__': _SAFE_BUILTINS.copy()}
        namespace.update(state)

        # Track which keys existed before execution
        keys_before = set(state.keys())

        # Separate the last expression for result capture
        last_expr_result = None
        stmts = tree.body
        last_is_expr = stmts and isinstance(stmts[-1], ast.Expr)

        # Capture stdout
        stdout_capture = io.StringIO()
        start_time = time.monotonic()

        try:
            if last_is_expr and len(stmts) > 1:
                # Execute all but last statement
                module_head = ast.Module(body=stmts[:-1], type_ignores=[])
                ast.fix_missing_locations(module_head)
                compiled_head = compile(module_head, '<sandbox>', 'exec')

                with redirect_stdout(stdout_capture):
                    exec(compiled_head, namespace)  # noqa: S102

                # Check timeout after head execution
                elapsed = time.monotonic() - start_time
                if elapsed > max_seconds:
                    return ExecutionResult(
                        output=stdout_capture.getvalue()[:max_output_chars],
                        result=None,
                        error=f"Execution timed out after {elapsed:.1f}s",
                        operations_count=node_count,
                    )

                # Evaluate last expression for its value
                expr_node = ast.Expression(body=stmts[-1].value)
                ast.fix_missing_locations(expr_node)
                compiled_expr = compile(expr_node, '<sandbox>', 'eval')

                with redirect_stdout(stdout_capture):
                    last_expr_result = eval(compiled_expr, namespace)  # noqa: S307

            elif last_is_expr and len(stmts) == 1:
                # Single expression - evaluate for result
                expr_node = ast.Expression(body=stmts[0].value)
                ast.fix_missing_locations(expr_node)
                compiled_expr = compile(expr_node, '<sandbox>', 'eval')

                with redirect_stdout(stdout_capture):
                    last_expr_result = eval(compiled_expr, namespace)  # noqa: S307

            else:
                # All statements, no expression result
                compiled = compile(tree, '<sandbox>', 'exec')
                with redirect_stdout(stdout_capture):
                    exec(compiled, namespace)  # noqa: S102

        except Exception as e:
            elapsed = time.monotonic() - start_time
            return ExecutionResult(
                output=stdout_capture.getvalue()[:max_output_chars],
                result=None,
                error=f"{type(e).__name__}: {e}",
                operations_count=node_count,
            )

        elapsed = time.monotonic() - start_time
        if elapsed > max_seconds:
            return ExecutionResult(
                output=stdout_capture.getvalue()[:max_output_chars],
                result=None,
                error=f"Execution timed out after {elapsed:.1f}s",
                operations_count=node_count,
            )

        # Sync namespace changes back to state
        variables_changed: list[str] = []
        for key, value in namespace.items():
            if key == '__builtins__':
                continue
            if key not in keys_before or state.get(key) is not value:
                state[key] = value
                variables_changed.append(key)

        # Track deletions
        keys_after = {k for k in namespace if k != '__builtins__'}
        for deleted_key in keys_before - keys_after:
            if deleted_key in state:
                del state[deleted_key]
                variables_changed.append(deleted_key)

        output = stdout_capture.getvalue()[:max_output_chars]

        return ExecutionResult(
            output=output,
            result=last_expr_result,
            error=None,
            variables_changed=variables_changed,
            operations_count=node_count,
        )

    def get_allowed_modules(self) -> list[str]:
        """Restricted executor does not allow any module imports."""
        return []
