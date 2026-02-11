"""Recursive Language Model (RLM) runner.

Drives iterative exec + llm_query cycles to achieve a goal:
1. LLM examines state + goal and generates code
2. Code runs in the sandbox
3. LLM examines results and decides if the goal is met
4. Repeat until goal met or max iterations reached
"""
import time
from typing import Any, Optional, TYPE_CHECKING

from scitrera_app_framework import get_logger, Variables

from ...config import (
    MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
    DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
    MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
    DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
)

if TYPE_CHECKING:
    from .default import DefaultContextEnvironmentService


_PLAN_SYSTEM_PROMPT = """You are an analytical reasoning agent working in a Python sandbox environment.
You have access to variables in the sandbox state and can execute Python code to transform data.

Your task is to achieve a GOAL by writing Python code. The code will be executed in a restricted
Python sandbox that supports: variable assignment, list/dict comprehensions, built-in functions
(len, sorted, sum, min, max, filter, map, etc.), string methods, and basic operations.

IMPORTANT RULES:
- Write ONLY Python code. Do not include markdown, explanations, or code fences.
- The code must be a single block that can be executed directly.
- Use variables from the current state to build toward the goal.
- Store meaningful intermediate results in named variables.
- If you believe the goal is achieved, set a variable called `_goal_achieved` to True.
- If you need to report a final answer, assign it to `_final_result`.

Current sandbox variables:
{state_summary}

Goal: {goal}"""

_PLAN_USER_PROMPT = """Based on the current state, write Python code to make progress toward the goal.

{iteration_context}

Write ONLY executable Python code (no markdown, no explanations):"""

_EVALUATE_SYSTEM_PROMPT = """You are evaluating whether a goal has been achieved based on the current
state of a Python sandbox environment.

Respond with EXACTLY one of:
- "ACHIEVED" if the goal is fully met
- "CONTINUE" if more work is needed
- "FAILED: <reason>" if the goal cannot be achieved

Goal: {goal}

Current state:
{state_summary}

Execution history:
{history}"""


def _summarize_state(state: dict[str, Any], max_chars: int = 5000) -> str:
    """Generate a concise summary of sandbox state for LLM context."""
    parts = []
    total_chars = 0
    for key, value in state.items():
        try:
            preview = repr(value)
        except Exception:
            preview = f"<{type(value).__name__}>"
        if len(preview) > 500:
            preview = preview[:500] + '...'

        line = f"  {key} ({type(value).__name__}): {preview}"
        if total_chars + len(line) > max_chars:
            parts.append(f"  ... ({len(state) - len(parts)} more variables)")
            break
        parts.append(line)
        total_chars += len(line)

    return '\n'.join(parts) if parts else '  (empty)'


class RLMRunner:
    """Drives iterative reasoning loops over sandbox environments."""

    def __init__(
        self,
        service: 'DefaultContextEnvironmentService',
        v: Variables,
    ):
        self._service = service
        self._v = v
        self.logger = get_logger(v, name=self.__class__.__name__)

        self._max_iterations = int(v.get(
            MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
            DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS,
        ))
        self._max_exec_seconds = int(v.get(
            MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
            DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS,
        ))

    async def run(
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
        """Run the RLM loop.

        Args:
            session_id: Session identifier
            goal: Natural language goal
            memory_query: Optional query to load initial memories
            memory_limit: Max memories to load
            max_iterations: Max iterations for this run
            variables: Variable names to include in LLM context
            result_var: Store final result in this variable
            detail_level: "minimal", "standard", or "verbose"

        Returns:
            Dict with result, iterations, trace, error, goal_achieved
        """
        effective_max = min(max_iterations, self._max_iterations)
        start_time = time.monotonic()
        trace: list[dict[str, Any]] = []

        self.logger.info(
            "RLM starting for session %s, goal: %s, max_iterations: %d",
            session_id, goal[:80], effective_max,
        )

        # Step 0: Load memories if requested
        if memory_query:
            load_result = await self._service.load(
                session_id=session_id,
                var='_memories',
                query=memory_query,
                limit=memory_limit,
            )
            if load_result.get('error'):
                return {
                    'result': None,
                    'iterations': 0,
                    'trace': [],
                    'error': f"Memory load failed: {load_result['error']}",
                    'goal_achieved': False,
                }
            self.logger.info(
                "RLM loaded %d memories for session %s",
                load_result.get('count', 0), session_id,
            )

        try:
            from ..llm import get_llm_service
            llm_service = get_llm_service(self._v)
        except Exception as e:
            return {
                'result': None,
                'iterations': 0,
                'trace': [],
                'error': f"LLM service not available: {e}",
                'goal_achieved': False,
            }

        goal_achieved = False
        final_result = None

        for iteration in range(effective_max):
            elapsed = time.monotonic() - start_time
            if elapsed > self._max_exec_seconds:
                self.logger.warning(
                    "RLM timed out for session %s after %.1fs",
                    session_id, elapsed,
                )
                trace.append({
                    'iteration': iteration,
                    'action': 'timeout',
                    'elapsed_seconds': round(elapsed, 1),
                })
                break

            iter_trace: dict[str, Any] = {'iteration': iteration}

            # Get current state summary
            status_result = await self._service.status(session_id)
            state = self._service._environments.get(session_id, {})
            state_summary = _summarize_state(state)

            # Build iteration context
            iteration_context = ""
            if iteration > 0 and trace:
                last = trace[-1]
                if last.get('exec_error'):
                    iteration_context = f"Previous iteration had an error: {last['exec_error']}"
                elif last.get('exec_output'):
                    iteration_context = f"Previous output: {last['exec_output'][:500]}"

            # Step 1: Ask LLM to generate code
            plan_system = _PLAN_SYSTEM_PROMPT.format(
                state_summary=state_summary,
                goal=goal,
            )
            plan_user = _PLAN_USER_PROMPT.format(
                iteration_context=iteration_context,
            )

            try:
                from ...models.llm import LLMRequest, LLMMessage, LLMRole
                plan_request = LLMRequest(
                    messages=[
                        LLMMessage(role=LLMRole.SYSTEM, content=plan_system),
                        LLMMessage(role=LLMRole.USER, content=plan_user),
                    ],
                    temperature=0.2,
                )
                plan_response = await llm_service.complete(plan_request)
                generated_code = plan_response.content.strip()
            except Exception as e:
                iter_trace['error'] = f"LLM plan generation failed: {e}"
                trace.append(iter_trace)
                self.logger.error("RLM plan generation failed: %s", e)
                break

            # Strip markdown code fences if present
            if generated_code.startswith('```'):
                lines = generated_code.split('\n')
                # Remove first and last fence lines
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                generated_code = '\n'.join(lines)

            iter_trace['generated_code'] = generated_code if detail_level != 'minimal' else '(omitted)'

            # Step 2: Execute the generated code
            exec_result = await self._service.execute(
                session_id=session_id,
                code=generated_code,
            )

            iter_trace['exec_output'] = exec_result.get('output', '')
            iter_trace['exec_error'] = exec_result.get('error')
            iter_trace['variables_changed'] = exec_result.get('variables_changed', [])

            # Check if sandbox set _goal_achieved or _final_result
            state = self._service._environments.get(session_id, {})
            if state.get('_goal_achieved'):
                goal_achieved = True
                final_result = state.get('_final_result')
                iter_trace['action'] = 'goal_achieved_by_code'
                trace.append(iter_trace)
                break

            if exec_result.get('error'):
                iter_trace['action'] = 'exec_error'
                trace.append(iter_trace)
                # Continue - LLM will see the error and adjust
                continue

            # Step 3: Ask LLM to evaluate progress
            history_summary = '\n'.join(
                f"  Iteration {t['iteration']}: "
                + (f"error={t.get('exec_error')}" if t.get('exec_error') else f"changed={t.get('variables_changed', [])}")
                for t in trace[-3:]  # Last 3 iterations for context
            )
            if iter_trace.get('variables_changed'):
                history_summary += f"\n  Current iteration: changed={iter_trace['variables_changed']}"

            eval_system = _EVALUATE_SYSTEM_PROMPT.format(
                goal=goal,
                state_summary=_summarize_state(state),
                history=history_summary or '  (first iteration)',
            )

            try:
                eval_request = LLMRequest(
                    messages=[
                        LLMMessage(role=LLMRole.SYSTEM, content=eval_system),
                        LLMMessage(role=LLMRole.USER, content="Is the goal achieved?"),
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                eval_response = await llm_service.complete(eval_request)
                evaluation = eval_response.content.strip()
            except Exception as e:
                iter_trace['eval_error'] = str(e)
                evaluation = "CONTINUE"

            iter_trace['evaluation'] = evaluation
            iter_trace['action'] = 'evaluated'
            trace.append(iter_trace)

            if evaluation.startswith('ACHIEVED'):
                goal_achieved = True
                final_result = state.get('_final_result')
                break
            elif evaluation.startswith('FAILED'):
                self.logger.info("RLM goal failed: %s", evaluation)
                break

        # Build final result
        if final_result is None and goal_achieved:
            # Try to find a meaningful result variable
            state = self._service._environments.get(session_id, {})
            final_result = state.get('_final_result') or state.get('result')

        # Store result if requested
        if result_var and final_result is not None:
            await self._service.inject(session_id, result_var, final_result)

        total_elapsed = time.monotonic() - start_time

        self.logger.info(
            "RLM completed for session %s: achieved=%s, iterations=%d, elapsed=%.1fs",
            session_id, goal_achieved, len(trace), total_elapsed,
        )

        # Clean trace for minimal detail level
        if detail_level == 'minimal':
            trace = [{'iteration': t['iteration'], 'action': t.get('action', 'unknown')} for t in trace]

        result_str = None
        if final_result is not None:
            try:
                result_str = repr(final_result)
                if len(result_str) > 10_000:
                    result_str = result_str[:10_000] + '...'
            except Exception:
                result_str = f"<{type(final_result).__name__}>"

        return {
            'result': result_str,
            'iterations': len(trace),
            'trace': trace,
            'error': None,
            'goal_achieved': goal_achieved,
        }
