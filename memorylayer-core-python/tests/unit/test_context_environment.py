"""Comprehensive tests for the context environment service.

Tests cover:
- RestrictedExecutor: AST-based safe execution
- SmolagentsExecutor: smolagents wrapper
- DefaultContextEnvironmentService: service logic
- API schemas: request/response validation
- Hooks: persistence hook interface
- RLM: reasoning loop runner (unit-level, no LLM)
"""
import asyncio
import json
import pytest

from memorylayer_server.services.context_environment.executors.base import (
    ExecutorProvider,
    ExecutionResult,
)
from memorylayer_server.services.context_environment.executors.restricted import (
    RestrictedExecutor,
)
from memorylayer_server.services.context_environment.hooks import (
    ContextPersistenceHook,
    NoOpPersistenceHook,
)
from memorylayer_server.services.context_environment.base import (
    ContextEnvironmentService,
    ContextEnvironmentServicePluginBase,
    EXT_CONTEXT_ENVIRONMENT_SERVICE,
)
from memorylayer_server.services.context_environment import (
    get_context_environment_service,
)
from memorylayer_server.services.context_environment.default import (
    DefaultContextEnvironmentService,
    DefaultContextEnvironmentServicePlugin,
    _safe_preview,
    _estimate_size,
    _memory_to_dict,
)
from memorylayer_server.services.context_environment.rlm import (
    RLMRunner,
    _summarize_state,
)
from memorylayer_server.config import (
    MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
    DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE,
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
    MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
    DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
    MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
    DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
    MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
    DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP,
    MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
    DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP,
)


# ============================================
# Fixtures
# ============================================

class MockVariables:
    """Minimal Variables mock for testing."""

    def __init__(self, overrides: dict = None):
        self._data = overrides or {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    def set_default_value(self, key, value):
        if key not in self._data:
            self._data[key] = value


class TrackingHook(ContextPersistenceHook):
    """Persistence hook that records calls for testing."""

    def __init__(self):
        self.state_changed_calls = []
        self.checkpoint_calls = []
        self.session_end_calls = []
        self.restore_calls = []

    async def on_state_changed(self, session_id: str, state: dict) -> None:
        self.state_changed_calls.append((session_id, dict(state)))

    async def on_checkpoint(self, session_id: str, state: dict) -> None:
        self.checkpoint_calls.append((session_id, dict(state)))

    async def on_session_end(self, session_id: str, state: dict) -> None:
        self.session_end_calls.append((session_id, dict(state)))

    async def on_session_restore(self, session_id: str) -> dict | None:
        self.restore_calls.append(session_id)
        return None


@pytest.fixture
def restricted_executor():
    return RestrictedExecutor()


@pytest.fixture
def mock_vars():
    return MockVariables()


@pytest.fixture
def tracking_hook():
    return TrackingHook()


@pytest.fixture
def service(mock_vars, restricted_executor, tracking_hook):
    return DefaultContextEnvironmentService(
        v=mock_vars,
        executor=restricted_executor,
        persistence_hook=tracking_hook,
    )


# ============================================
# Config Constants Tests
# ============================================

class TestConfigConstants:
    """Test that all config constants are properly defined."""

    def test_service_config(self):
        assert MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE == 'MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE'
        assert DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE == 'default'

    def test_executor_config(self):
        assert MEMORYLAYER_CONTEXT_EXECUTOR == 'MEMORYLAYER_CONTEXT_EXECUTOR'
        assert DEFAULT_MEMORYLAYER_CONTEXT_EXECUTOR == 'smolagents'

    def test_limits_config(self):
        assert DEFAULT_MEMORYLAYER_CONTEXT_MAX_OPERATIONS == 1_000_000
        assert DEFAULT_MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS == 30
        assert DEFAULT_MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS == 50_000
        assert DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS == 4096
        assert DEFAULT_MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES == 256 * 1024 * 1024

    def test_rlm_config(self):
        assert DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS == 10
        assert DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS == 120

    def test_cap_config(self):
        assert DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP == 0
        assert DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP == 0


# ============================================
# ExecutionResult Tests
# ============================================

class TestExecutionResult:
    """Test the ExecutionResult dataclass."""

    def test_default_fields(self):
        result = ExecutionResult(output='hello', result=42, error=None)
        assert result.output == 'hello'
        assert result.result == 42
        assert result.error is None
        assert result.variables_changed == []
        assert result.operations_count == 0

    def test_with_all_fields(self):
        result = ExecutionResult(
            output='out',
            result='val',
            error='err',
            variables_changed=['x', 'y'],
            operations_count=10,
        )
        assert result.variables_changed == ['x', 'y']
        assert result.operations_count == 10


# ============================================
# RestrictedExecutor Tests
# ============================================

class TestRestrictedExecutor:
    """Test the AST-based restricted executor."""

    async def test_empty_code(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('', state)
        assert result.error is None
        assert result.output == ''
        assert result.result is None

    async def test_whitespace_only(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('   \n  \n  ', state)
        assert result.error is None

    async def test_simple_assignment(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = 42', state)
        assert result.error is None
        assert state['x'] == 42
        assert 'x' in result.variables_changed

    async def test_multiple_assignments(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = 1\ny = 2\nz = 3', state)
        assert result.error is None
        assert state['x'] == 1
        assert state['y'] == 2
        assert state['z'] == 3

    async def test_expression_result(self, restricted_executor):
        state = {'x': 10}
        result = await restricted_executor.execute('x + 5', state)
        assert result.error is None
        assert result.result == 15

    async def test_assignment_then_expression(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = 10\ny = 20\nx + y', state)
        assert result.error is None
        assert result.result == 30

    async def test_list_comprehension(self, restricted_executor):
        state = {'data': [1, 2, 3, 4, 5]}
        result = await restricted_executor.execute(
            'evens = [x for x in data if x % 2 == 0]', state
        )
        assert result.error is None
        assert state['evens'] == [2, 4]

    async def test_dict_comprehension(self, restricted_executor):
        state = {'keys': ['a', 'b', 'c']}
        result = await restricted_executor.execute(
            'mapping = {k: i for i, k in enumerate(keys)}', state
        )
        assert result.error is None
        assert state['mapping'] == {'a': 0, 'b': 1, 'c': 2}

    async def test_builtin_functions(self, restricted_executor):
        state = {'data': [3, 1, 4, 1, 5]}
        result = await restricted_executor.execute('result = sorted(data)', state)
        assert result.error is None
        assert state['result'] == [1, 1, 3, 4, 5]

    async def test_len_min_max_sum(self, restricted_executor):
        state = {'nums': [10, 20, 30]}
        result = await restricted_executor.execute(
            'total = sum(nums)\ncount = len(nums)\nlo = min(nums)\nhi = max(nums)',
            state,
        )
        assert result.error is None
        assert state['total'] == 60
        assert state['count'] == 3
        assert state['lo'] == 10
        assert state['hi'] == 30

    async def test_string_methods(self, restricted_executor):
        state = {'text': 'Hello World'}
        result = await restricted_executor.execute(
            'upper = text.upper()\nlower = text.lower()\nwords = text.split()',
            state,
        )
        assert result.error is None
        assert state['upper'] == 'HELLO WORLD'
        assert state['lower'] == 'hello world'
        assert state['words'] == ['Hello', 'World']

    async def test_subscript_and_slice(self, restricted_executor):
        state = {'items': [10, 20, 30, 40, 50]}
        result = await restricted_executor.execute(
            'first = items[0]\nlast = items[-1]\nmid = items[1:4]',
            state,
        )
        assert result.error is None
        assert state['first'] == 10
        assert state['last'] == 50
        assert state['mid'] == [20, 30, 40]

    async def test_dict_access(self, restricted_executor):
        state = {'d': {'name': 'test', 'value': 42}}
        result = await restricted_executor.execute('name = d["name"]', state)
        assert result.error is None
        assert state['name'] == 'test'

    async def test_boolean_logic(self, restricted_executor):
        state = {'a': True, 'b': False}
        result = await restricted_executor.execute(
            'c = a and b\nd = a or b\ne = not b',
            state,
        )
        assert result.error is None
        assert state['c'] is False
        assert state['d'] is True
        assert state['e'] is True

    async def test_comparison(self, restricted_executor):
        state = {'x': 5}
        result = await restricted_executor.execute(
            'gt = x > 3\neq = x == 5\nlt = x < 2',
            state,
        )
        assert result.error is None
        assert state['gt'] is True
        assert state['eq'] is True
        assert state['lt'] is False

    async def test_if_expression(self, restricted_executor):
        state = {'x': 10}
        result = await restricted_executor.execute(
            'label = "big" if x > 5 else "small"',
            state,
        )
        assert result.error is None
        assert state['label'] == 'big'

    async def test_del_statement(self, restricted_executor):
        state = {'x': 1, 'y': 2}
        result = await restricted_executor.execute('del x', state)
        assert result.error is None
        assert 'x' not in state
        assert 'y' in state
        assert 'x' in result.variables_changed

    async def test_augmented_assignment(self, restricted_executor):
        state = {'x': 10}
        result = await restricted_executor.execute('x += 5', state)
        assert result.error is None
        assert state['x'] == 15

    async def test_print_capture(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('print("hello world")', state)
        assert result.error is None
        assert 'hello world' in result.output

    async def test_state_persistence(self, restricted_executor):
        state = {}
        await restricted_executor.execute('x = 1', state)
        await restricted_executor.execute('y = x + 1', state)
        result = await restricted_executor.execute('x + y', state)
        assert result.result == 3

    # --- Blocked operations ---

    async def test_import_blocked(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('import os', state)
        assert result.error is not None
        assert 'Disallowed' in result.error or 'Import' in result.error

    async def test_for_loop_blocked(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute(
            'total = 0\nfor i in range(5):\n    total += i', state
        )
        assert result.error is not None

    async def test_while_loop_blocked(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute(
            'x = 0\nwhile x < 5:\n    x += 1', state
        )
        assert result.error is not None

    async def test_function_def_blocked(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('def foo(): return 42', state)
        assert result.error is not None

    async def test_class_def_blocked(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('class Foo: pass', state)
        assert result.error is not None

    async def test_exec_eval_blocked(self, restricted_executor):
        state = {}
        # exec/eval would require Call nodes with those names
        # but they're not in safe builtins, so they'd fail at runtime
        result = await restricted_executor.execute('exec("x = 1")', state)
        assert result.error is not None

    async def test_syntax_error(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = ', state)
        assert result.error is not None
        assert 'Syntax error' in result.error

    async def test_runtime_error(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = 1 / 0', state)
        assert result.error is not None
        assert 'ZeroDivision' in result.error

    async def test_name_error(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute('x = undefined_var', state)
        assert result.error is not None
        assert 'NameError' in result.error

    async def test_output_truncation(self, restricted_executor):
        state = {}
        result = await restricted_executor.execute(
            'print("x" * 200)', state, max_output_chars=50
        )
        assert result.error is None
        assert len(result.output) <= 50

    async def test_allowed_modules_empty(self, restricted_executor):
        assert restricted_executor.get_allowed_modules() == []

    async def test_if_statement(self, restricted_executor):
        state = {'x': 10}
        result = await restricted_executor.execute(
            'if x > 5:\n    label = "big"\n',
            state,
        )
        assert result.error is None
        assert state['label'] == 'big'


# ============================================
# SmolagentsExecutor Tests
# ============================================

class TestSmolagentsExecutor:
    """Test the smolagents executor wrapper."""

    @pytest.fixture
    def smolagents_executor(self):
        from memorylayer_server.services.context_environment.executors.smolagents_executor import (
            SmolagentsExecutor,
        )
        return SmolagentsExecutor()

    async def test_simple_assignment(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute('x = 42', state)
        assert result.error is None
        assert state['x'] == 42

    async def test_expression_result(self, smolagents_executor):
        state = {'x': 10}
        result = await smolagents_executor.execute('x + 5', state)
        assert result.error is None
        assert result.result == 15

    async def test_builtin_sum(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute(
            'data = [1, 2, 3]\ntotal = sum(data)', state
        )
        assert result.error is None
        assert state['total'] == 6

    async def test_json_import(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute(
            'import json\nresult = json.dumps({"key": "value"})',
            state,
        )
        assert result.error is None
        assert state['result'] == '{"key": "value"}'

    async def test_forbidden_import(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute('import os', state)
        assert result.error is not None

    async def test_state_persistence(self, smolagents_executor):
        state = {}
        await smolagents_executor.execute('x = 10', state)
        await smolagents_executor.execute('y = x * 2', state)
        result = await smolagents_executor.execute('x + y', state)
        assert result.result == 30

    async def test_variables_changed_tracking(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute('a = 1\nb = 2', state)
        assert 'a' in result.variables_changed
        assert 'b' in result.variables_changed

    async def test_allowed_modules(self, smolagents_executor):
        modules = smolagents_executor.get_allowed_modules()
        assert 'json' in modules
        assert 'math' in modules
        assert 'collections' in modules
        assert 'datetime' in modules
        assert 'functools' in modules

    async def test_empty_code(self, smolagents_executor):
        state = {}
        result = await smolagents_executor.execute('', state)
        assert result.error is None

    async def test_comprehension(self, smolagents_executor):
        state = {'nums': [1, 2, 3, 4, 5]}
        result = await smolagents_executor.execute(
            'doubled = [x * 2 for x in nums]', state
        )
        assert result.error is None
        assert state['doubled'] == [2, 4, 6, 8, 10]


# ============================================
# Hooks Tests
# ============================================

class TestHooks:
    """Test persistence hook interface."""

    async def test_noop_hook(self):
        hook = NoOpPersistenceHook()
        # All methods should be callable without error
        await hook.on_state_changed('sess', {})
        await hook.on_checkpoint('sess', {})
        await hook.on_session_end('sess', {})
        result = await hook.on_session_restore('sess')
        assert result is None

    async def test_tracking_hook(self, tracking_hook):
        await tracking_hook.on_state_changed('s1', {'x': 1})
        await tracking_hook.on_checkpoint('s1', {'x': 1})
        await tracking_hook.on_session_end('s1', {'x': 1})
        await tracking_hook.on_session_restore('s1')

        assert len(tracking_hook.state_changed_calls) == 1
        assert tracking_hook.state_changed_calls[0][0] == 's1'
        assert len(tracking_hook.checkpoint_calls) == 1
        assert len(tracking_hook.session_end_calls) == 1
        assert tracking_hook.restore_calls == ['s1']


# ============================================
# Service ABC Tests
# ============================================

class TestServiceABC:
    """Test service ABC and plugin base."""

    def test_extension_point_constant(self):
        assert EXT_CONTEXT_ENVIRONMENT_SERVICE == 'memorylayer-context-environment-service'

    def test_plugin_base_name(self):
        class TestPlugin(ContextEnvironmentServicePluginBase):
            PROVIDER_NAME = 'test'
        plugin = TestPlugin()
        assert plugin.name() == 'memorylayer-context-environment-service|test'

    def test_plugin_base_extension_point(self):
        class TestPlugin(ContextEnvironmentServicePluginBase):
            PROVIDER_NAME = 'test'
        plugin = TestPlugin()
        v = MockVariables()
        assert plugin.extension_point_name(v) == EXT_CONTEXT_ENVIRONMENT_SERVICE


# ============================================
# DefaultContextEnvironmentService Tests
# ============================================

class TestDefaultContextEnvironmentService:
    """Test the default service implementation."""

    async def test_execute_basic(self, service):
        result = await service.execute('s1', 'x = 42')
        assert result['error'] is None
        assert 'x' in result['variables_changed']

    async def test_execute_expression(self, service):
        await service.execute('s1', 'x = 10')
        result = await service.execute('s1', 'x + 5')
        assert result['error'] is None
        assert '15' in str(result['result'])

    async def test_execute_result_var(self, service):
        await service.execute('s1', 'x = [1, 2, 3]')
        result = await service.execute('s1', 'len(x)', result_var='count')
        assert result['error'] is None
        assert 'count' in result['variables_changed']

    async def test_execute_return_result_false(self, service):
        result = await service.execute('s1', '42', return_result=False)
        assert result['result'] is None

    async def test_execute_error(self, service):
        result = await service.execute('s1', 'x = 1 / 0')
        assert result['error'] is not None
        assert 'ZeroDivision' in result['error']

    async def test_inspect_all(self, service):
        await service.execute('s1', 'x = 1\ny = "hello"')
        result = await service.inspect('s1')
        assert result['variable_count'] == 2
        assert 'x' in result['variables']
        assert 'y' in result['variables']

    async def test_inspect_specific_variable(self, service):
        await service.execute('s1', 'x = 42')
        result = await service.inspect('s1', variable='x')
        assert result['type'] == 'int'
        assert '42' in result['preview']

    async def test_inspect_nonexistent_variable(self, service):
        await service._init_environment('s1')
        result = await service.inspect('s1', variable='missing')
        assert 'error' in result

    async def test_inspect_creates_environment(self, service):
        result = await service.inspect('new_session')
        assert result['variable_count'] == 0

    async def test_inject_value(self, service):
        result = await service.inject('s1', 'data', [1, 2, 3])
        assert result['variable'] == 'data'
        assert result['type'] == 'list'

    async def test_inject_json(self, service):
        result = await service.inject(
            's1', 'config', '{"key": "value"}', parse_json=True
        )
        assert result['type'] == 'dict'

    async def test_inject_invalid_json(self, service):
        result = await service.inject(
            's1', 'bad', 'not json{', parse_json=True
        )
        assert 'error' in result

    async def test_inject_then_execute(self, service):
        await service.inject('s1', 'data', [10, 20, 30])
        result = await service.execute('s1', 'total = sum(data)')
        assert result['error'] is None

    async def test_status_no_environment(self, service):
        result = await service.status('nonexistent')
        assert result['exists'] is False

    async def test_status_with_environment(self, service):
        await service.execute('s1', 'x = 1\ny = 2')
        result = await service.status('s1')
        assert result['exists'] is True
        assert result['variable_count'] == 2
        assert 'x' in result['variables']

    async def test_cleanup(self, service):
        await service.execute('s1', 'x = 42')
        await service.cleanup_environment('s1')
        result = await service.status('s1')
        assert result['exists'] is False

    async def test_cleanup_nonexistent(self, service):
        # Should not raise
        await service.cleanup_environment('nonexistent')

    async def test_session_isolation(self, service):
        await service.execute('s1', 'x = 1')
        await service.execute('s2', 'x = 2')

        r1 = await service.execute('s1', 'x')
        r2 = await service.execute('s2', 'x')

        assert '1' in str(r1['result'])
        assert '2' in str(r2['result'])

    async def test_persistence_hook_state_changed(self, service, tracking_hook):
        await service.execute('s1', 'x = 42')
        assert len(tracking_hook.state_changed_calls) >= 1
        assert tracking_hook.state_changed_calls[0][0] == 's1'

    async def test_persistence_hook_no_call_on_error(self, service, tracking_hook):
        initial_count = len(tracking_hook.state_changed_calls)
        await service.execute('s1', 'x = 1 / 0')
        assert len(tracking_hook.state_changed_calls) == initial_count

    async def test_persistence_hook_session_end(self, service, tracking_hook):
        await service.execute('s1', 'x = 1')
        await service.cleanup_environment('s1')
        assert len(tracking_hook.session_end_calls) == 1
        assert tracking_hook.session_end_calls[0][0] == 's1'

    async def test_metadata_tracking(self, service):
        await service.execute('s1', 'x = 1')
        await service.execute('s1', 'y = 2')
        meta = service._env_metadata['s1']
        assert meta['exec_count'] == 2
        assert 'last_exec_at' in meta
        assert 'created_at' in meta

    async def test_query_without_llm(self, service):
        await service.inject('s1', 'data', [1, 2, 3])
        result = await service.query('s1', 'Summarize the data', ['data'])
        assert 'error' in result
        # LLM not available in test environment

    async def test_load_without_memory_service(self, service):
        result = await service.load('s1', 'memories', 'test query')
        assert result.get('error') is not None or result.get('count') == 0

    async def test_rlm_without_llm(self, service):
        result = await service.rlm('s1', 'test goal')
        assert result.get('error') is not None
        assert 'LLM' in result['error']

    async def test_hard_cap_enforcement(self):
        v = MockVariables({MEMORYLAYER_CONTEXT_EXEC_HARD_CAP: 2})
        svc = DefaultContextEnvironmentService(
            v=v, executor=RestrictedExecutor()
        )
        await svc.execute('s1', 'x = 1')
        await svc.execute('s1', 'y = 2')
        result = await svc.execute('s1', 'z = 3')
        assert result['error'] is not None
        assert 'cap' in result['error'].lower()


# ============================================
# Helper Function Tests
# ============================================

class TestHelperFunctions:
    """Test utility/helper functions."""

    def test_safe_preview_short(self):
        assert _safe_preview(42) == '42'

    def test_safe_preview_long(self):
        long_list = list(range(1000))
        preview = _safe_preview(long_list, max_chars=50)
        assert len(preview) <= 53  # 50 + '...'
        assert preview.endswith('...')

    def test_safe_preview_unprintable(self):
        class BadRepr:
            def __repr__(self):
                raise RuntimeError("bad")
        preview = _safe_preview(BadRepr())
        assert 'BadRepr' in preview

    def test_estimate_size(self):
        assert _estimate_size(42) > 0
        assert _estimate_size([1, 2, 3]) > 0
        assert _estimate_size("hello") > 0

    def test_summarize_state(self):
        state = {'x': 42, 'data': [1, 2, 3]}
        summary = _summarize_state(state)
        assert 'x' in summary
        assert 'data' in summary
        assert 'int' in summary
        assert 'list' in summary

    def test_summarize_state_empty(self):
        summary = _summarize_state({})
        assert 'empty' in summary

    def test_summarize_state_truncation(self):
        state = {f'var_{i}': i for i in range(100)}
        summary = _summarize_state(state, max_chars=200)
        assert 'more variables' in summary


# ============================================
# Schema Tests
# ============================================

class TestSchemas:
    """Test API request/response schema validation."""

    def test_execute_request(self):
        from memorylayer_server.api.v1.schemas import ContextExecuteRequest

        req = ContextExecuteRequest(code='x = 42')
        assert req.code == 'x = 42'
        assert req.result_var is None
        assert req.return_result is True
        assert req.max_return_chars == 10_000

    def test_execute_request_min_length(self):
        from memorylayer_server.api.v1.schemas import ContextExecuteRequest

        with pytest.raises(Exception):
            ContextExecuteRequest(code='')

    def test_execute_response(self):
        from memorylayer_server.api.v1.schemas import ContextExecuteResponse

        resp = ContextExecuteResponse(output='hello', result='42')
        assert resp.output == 'hello'
        assert resp.result == '42'
        assert resp.error is None
        assert resp.variables_changed == []

    def test_load_request(self):
        from memorylayer_server.api.v1.schemas import ContextLoadRequest

        req = ContextLoadRequest(var='memories', query='test')
        assert req.var == 'memories'
        assert req.query == 'test'
        assert req.limit == 50
        assert req.include_embeddings is False

    def test_inject_request(self):
        from memorylayer_server.api.v1.schemas import ContextInjectRequest

        req = ContextInjectRequest(key='data', value=[1, 2, 3])
        assert req.key == 'data'
        assert req.value == [1, 2, 3]
        assert req.parse_json is False

    def test_query_request(self):
        from memorylayer_server.api.v1.schemas import ContextQueryRequest

        req = ContextQueryRequest(prompt='Summarize', variables=['data', 'config'])
        assert req.prompt == 'Summarize'
        assert req.variables == ['data', 'config']

    def test_rlm_request(self):
        from memorylayer_server.api.v1.schemas import ContextRLMRequest

        req = ContextRLMRequest(goal='Analyze trends')
        assert req.goal == 'Analyze trends'
        assert req.max_iterations == 10
        assert req.detail_level == 'standard'

    def test_rlm_response(self):
        from memorylayer_server.api.v1.schemas import ContextRLMResponse

        resp = ContextRLMResponse(
            result='done',
            iterations=3,
            goal_achieved=True,
        )
        assert resp.result == 'done'
        assert resp.iterations == 3
        assert resp.goal_achieved is True
        assert resp.trace == []

    def test_status_response(self):
        from memorylayer_server.api.v1.schemas import ContextStatusResponse

        resp = ContextStatusResponse(exists=True, variable_count=5)
        assert resp.exists is True
        assert resp.variable_count == 5

    def test_inspect_response(self):
        from memorylayer_server.api.v1.schemas import ContextInspectResponse

        resp = ContextInspectResponse(
            variable='x', type='int', preview='42', size_bytes=28
        )
        assert resp.variable == 'x'
        assert resp.type == 'int'


# ============================================
# API Router Tests
# ============================================

class TestAPIRouter:
    """Test API router registration."""

    def test_router_has_all_routes(self):
        from memorylayer_server.api.v1.context_environment import router

        paths = [r.path for r in router.routes]
        assert '/v1/context/execute' in paths
        assert '/v1/context/inspect' in paths
        assert '/v1/context/load' in paths
        assert '/v1/context/inject' in paths
        assert '/v1/context/query' in paths
        assert '/v1/context/rlm' in paths
        assert '/v1/context/status' in paths
        assert '/v1/context/cleanup' in paths

    def test_plugin_registration(self):
        from memorylayer_server.api.v1.context_environment import ContextEnvironmentAPIPlugin

        plugin = ContextEnvironmentAPIPlugin()
        v = MockVariables()
        assert plugin.extension_point_name(v) == 'memorylayer-server-api-routers'
        assert plugin.is_multi_extension(v) is True
        assert plugin.is_enabled(v) is False  # Multi-extension pattern


# ============================================
# Plugin Tests
# ============================================

class TestDefaultPlugin:
    """Test the default service plugin."""

    def test_plugin_provider_name(self):
        plugin = DefaultContextEnvironmentServicePlugin()
        assert plugin.PROVIDER_NAME == 'default'
        assert 'default' in plugin.name()

    def test_plugin_extension_point(self):
        plugin = DefaultContextEnvironmentServicePlugin()
        v = MockVariables()
        assert plugin.extension_point_name(v) == EXT_CONTEXT_ENVIRONMENT_SERVICE

    def test_plugin_on_registration(self):
        plugin = DefaultContextEnvironmentServicePlugin()
        v = MockVariables()
        plugin.on_registration(v)
        assert v.get(MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE) == 'default'
