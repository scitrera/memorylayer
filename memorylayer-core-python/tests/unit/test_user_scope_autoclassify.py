"""Tests for the Slice 2 preference-vs-episodic autoclassifier.

Slice 1 (test_user_scope_write.py) covers the explicit scope=USER routing and
the PREFERENCE/DIRECTIVE subtype map. Slice 2 layers AUTOMATIC classification on
top, feeding the SAME routing seam (_route_user_scope). These tests cover:

  * classifier ON + classified-preference (>= threshold) + user_id -> USER scope
  * below-threshold confidence -> stays workspace
  * classifier OFF (the default) -> NEVER reroutes (byte-identical to Slice 1)
  * classifier error/timeout/unparseable -> workspace scope, no raise (fail-safe)
  * explicit scope=USER and scope=WORKSPACE BOTH bypass the classifier
  * no user_id + classified-preference -> scope DOWN to workspace (no global row)
  * episodic content -> NOT routed to user scope (false-positive guard)
  * the OSS HeuristicScopeClassifier itself (deterministic)
"""

import pytest
import pytest_asyncio

from memorylayer_server.config import (
    DEFAULT_TENANT_ID,
    GLOBAL_USER_WORKSPACE_ID,
)
from memorylayer_server.models import (
    MemoryScope,
    MemoryType,
    RememberInput,
    Workspace,
)
from memorylayer_server.services.memory.scope_classifier import (
    HeuristicScopeClassifier,
    NoopScopeClassifier,
    ScopeClassification,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def test_workspace(storage_backend, unique_workspace_id):
    workspace = Workspace(
        id=unique_workspace_id,
        tenant_id=DEFAULT_TENANT_ID,
        name="Autoclassify Test Workspace",
    )
    return await storage_backend.create_workspace(workspace)


@pytest_asyncio.fixture
async def global_user_workspace(storage_backend):
    existing = await storage_backend.get_workspace(GLOBAL_USER_WORKSPACE_ID)
    if existing:
        return existing
    workspace = Workspace(
        id=GLOBAL_USER_WORKSPACE_ID,
        tenant_id=DEFAULT_TENANT_ID,
        name="Global User Workspace",
    )
    return await storage_backend.create_workspace(workspace)


class _StubClassifier:
    """A fake classifier returning a fixed verdict (or raising) for tests."""

    def __init__(self, verdict=None, raises=False):
        self._verdict = verdict
        self._raises = raises
        self.calls = 0

    async def classify(self, content: str) -> ScopeClassification:
        self.calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._verdict


@pytest.fixture
def autoclassify_on(memory_service):
    """Enable the autoclassify master knob and restore afterwards (the
    memory_service is a shared DI singleton)."""
    orig_enabled = memory_service.user_scope_autoclassify_enabled
    orig_threshold = memory_service.user_scope_autoclassify_threshold
    orig_classifier = memory_service.scope_classifier
    memory_service.user_scope_autoclassify_enabled = True
    try:
        yield memory_service
    finally:
        memory_service.user_scope_autoclassify_enabled = orig_enabled
        memory_service.user_scope_autoclassify_threshold = orig_threshold
        memory_service.scope_classifier = orig_classifier


# --------------------------------------------------------------------------
# ON + classified-preference -> USER scope
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classified_preference_routes_to_user_scope(
    autoclassify_on, test_workspace, global_user_workspace
):
    """Classifier ON, returns user-preference at/above threshold, user_id set
    -> routed to _global_user with origin_workspace_id provenance."""
    svc = autoclassify_on
    svc.user_scope_autoclassify_threshold = 0.85
    svc.scope_classifier = _StubClassifier(
        ScopeClassification(True, 0.95, "durable preference")
    )

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Ivan classified-pref likes terse responses with code samples",
            type=MemoryType.SEMANTIC,
            user_id="ivan@example.com",
        ),
    )

    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    assert memory.user_id == "ivan@example.com"
    assert memory.metadata.get("origin_workspace_id") == test_workspace.id


@pytest.mark.asyncio
async def test_below_threshold_stays_workspace(
    autoclassify_on, test_workspace, global_user_workspace
):
    """Classifier returns a preference but BELOW the threshold -> workspace."""
    svc = autoclassify_on
    svc.user_scope_autoclassify_threshold = 0.85
    svc.scope_classifier = _StubClassifier(
        ScopeClassification(True, 0.50, "weak signal")
    )

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Below-threshold maybe-a-preference content",
            type=MemoryType.SEMANTIC,
            user_id="judy@example.com",
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


# --------------------------------------------------------------------------
# OFF (default) -> never reroutes; classifier is never even consulted
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_off_by_default_never_reroutes(
    memory_service, test_workspace, global_user_workspace
):
    """The master knob defaults OFF: a would-be preference stays workspace-scoped
    and the classifier is NOT consulted (zero added latency)."""
    assert memory_service.user_scope_autoclassify_enabled is False

    stub = _StubClassifier(ScopeClassification(True, 1.0, "would route"))
    orig = memory_service.scope_classifier
    memory_service.scope_classifier = stub
    try:
        memory = await memory_service.remember(
            workspace_id=test_workspace.id,
            input=RememberInput(
                content="Knob-off I always prefer dark mode in every editor",
                type=MemoryType.SEMANTIC,
                user_id="ken@example.com",
            ),
        )
        assert memory.workspace_id == test_workspace.id
        assert "origin_workspace_id" not in memory.metadata
        assert stub.calls == 0  # gated behind the flag -> never invoked
    finally:
        memory_service.scope_classifier = orig


# --------------------------------------------------------------------------
# Fail-safe: classifier error -> workspace, no raise
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_error_degrades_to_workspace(
    autoclassify_on, test_workspace, global_user_workspace
):
    """A classifier that raises must NOT block the write; the memory lands in
    the origin workspace (fail-safe)."""
    svc = autoclassify_on
    svc.scope_classifier = _StubClassifier(raises=True)

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Classifier-error content that should still persist",
            type=MemoryType.SEMANTIC,
            user_id="laura@example.com",
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


# --------------------------------------------------------------------------
# Explicit scope ALWAYS wins (bypasses classifier)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_workspace_bypasses_classifier(
    autoclassify_on, test_workspace, global_user_workspace
):
    """Explicit scope=WORKSPACE keeps the memory local even when the classifier
    would say user-preference, and the classifier is not consulted."""
    svc = autoclassify_on
    stub = _StubClassifier(ScopeClassification(True, 1.0, "would route"))
    svc.scope_classifier = stub

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Explicit-workspace I always prefer vim everywhere",
            type=MemoryType.SEMANTIC,
            user_id="mike@example.com",
            scope=MemoryScope.WORKSPACE,
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata
    assert stub.calls == 0


@pytest.mark.asyncio
async def test_explicit_user_bypasses_classifier(
    autoclassify_on, test_workspace, global_user_workspace
):
    """Explicit scope=USER routes to user scope without consulting the
    classifier (even one that would say NOT a preference)."""
    svc = autoclassify_on
    stub = _StubClassifier(ScopeClassification(False, 0.0, "would NOT route"))
    svc.scope_classifier = stub

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="Explicit-user this is an episodic-sounding fact",
            type=MemoryType.SEMANTIC,
            user_id="nina@example.com",
            scope=MemoryScope.USER,
        ),
    )
    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    assert stub.calls == 0


# --------------------------------------------------------------------------
# No user_id + classified-preference -> scope DOWN (never an unfilterable row)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classified_preference_without_user_id_scopes_down(
    autoclassify_on, test_workspace
):
    """Classifier says user-preference but there is no user_id -> the memory is
    scoped DOWN to the origin workspace (not rejected, not an unfilterable
    global row)."""
    svc = autoclassify_on
    svc.user_scope_autoclassify_threshold = 0.85
    svc.scope_classifier = _StubClassifier(
        ScopeClassification(True, 0.99, "preference but no owner")
    )

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="No-user-id classified preference content",
            type=MemoryType.SEMANTIC,
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


# --------------------------------------------------------------------------
# Episodic content -> NOT routed (false-positive guard), via the OSS heuristic
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_episodic_content_not_routed_oss_heuristic(
    autoclassify_on, test_workspace, global_user_workspace
):
    """With the real OSS heuristic and a threshold it could clear, an episodic
    project decision is NOT routed to user scope (the heuristic vetoes it)."""
    svc = autoclassify_on
    svc.user_scope_autoclassify_threshold = 0.5  # below heuristic match conf
    svc.scope_classifier = HeuristicScopeClassifier()

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="We decided to use Postgres in project X for the analytics store",
            type=MemoryType.SEMANTIC,
            user_id="oscar@example.com",
        ),
    )
    assert memory.workspace_id == test_workspace.id
    assert "origin_workspace_id" not in memory.metadata


@pytest.mark.asyncio
async def test_durable_preference_routed_oss_heuristic(
    autoclassify_on, test_workspace, global_user_workspace
):
    """With the real OSS heuristic and a threshold at/below its match
    confidence, a clear durable first-person preference routes to user scope."""
    svc = autoclassify_on
    svc.user_scope_autoclassify_threshold = 0.5
    svc.scope_classifier = HeuristicScopeClassifier()

    memory = await svc.remember(
        workspace_id=test_workspace.id,
        input=RememberInput(
            content="I always prefer concise answers with bullet points",
            type=MemoryType.SEMANTIC,
            user_id="peggy@example.com",
        ),
    )
    assert memory.workspace_id == GLOBAL_USER_WORKSPACE_ID
    assert memory.user_id == "peggy@example.com"


# --------------------------------------------------------------------------
# The OSS HeuristicScopeClassifier in isolation (deterministic)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "I prefer dark mode",
        "I always respond in English",
        "My favorite editor is vim",
        "Please always answer concisely",
        "Call me Dr. Smith",
        "I don't like verbose explanations",
    ],
)
async def test_heuristic_matches_durable_preferences(content):
    result = await HeuristicScopeClassifier().classify(content)
    assert result.is_user_preference is True
    assert result.confidence > 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "We decided to use Postgres in project X",
        "The build uses cmake",
        "Our team agreed to ship on Friday",
        "The service returned a 500 error today",
        "",
        "   ",
    ],
)
async def test_heuristic_rejects_episodic_and_empty(content):
    result = await HeuristicScopeClassifier().classify(content)
    assert result.is_user_preference is False
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_heuristic_veto_beats_preference_cue():
    """An episodic veto wins even when a preference verb is also present."""
    # "we decided" veto present alongside "use"
    result = await HeuristicScopeClassifier().classify(
        "We decided we always use tabs in the project repo"
    )
    assert result.is_user_preference is False


@pytest.mark.asyncio
async def test_noop_classifier_never_routes():
    result = await NoopScopeClassifier().classify("I always prefer dark mode")
    assert result.is_user_preference is False
    assert result.confidence == 0.0
