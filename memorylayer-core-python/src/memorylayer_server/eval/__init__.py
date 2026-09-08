"""MemoryLayer retrieval evaluation harness.

A small, dependency-light framework for measuring recall quality against a fixed
corpus + qrels and gating CI on the result. Run it with::

    python -m memorylayer_server.eval gate        # uses the bundled fixture set
    memorylayer-eval gate --json                  # console-script equivalent

Design ported from gbrain's eval framework (MIT). The metric and gate layers are
pure and unit-testable; the harness drives the real ``MemoryService.recall()``
pipeline so ranking changes show up in the numbers.
"""

from .gate import evaluate_gate
from .harness import DEFAULT_CORPUS, DEFAULT_QRELS, evaluate, run_eval
from .models import EvalReport, GateResult, Qrel, Thresholds

__all__ = (
    "evaluate",
    "run_eval",
    "evaluate_gate",
    "EvalReport",
    "GateResult",
    "Qrel",
    "Thresholds",
    "DEFAULT_CORPUS",
    "DEFAULT_QRELS",
)
