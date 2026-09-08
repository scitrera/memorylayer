"""Tests for create-time entity-metadata routing through the ExtractionService.

These pin down the root fix: ``_maybe_add_entity_metadata`` must source its
speaker/entities/entity_types from ``self.extraction_service`` (the pluggable
provider, e.g. GLiNER2) — not the hardcoded regex util — so the selected
provider drives BOTH the entity-anchor recall channel and registry accretion.

Backward-compat is the hard constraint: when the provider is the regex-backed
``DefaultExtractionService`` (whose ``extract_entities`` == the regex util), the
folded metadata must be byte-identical to what the regex util produces today
(golden test). ``_accrete_entities`` typing is also covered: typed providers map
each entity onto its ``EntityType``; absent types fall back to the
speaker-is-PERSON / else-CONCEPT heuristic.
"""

import uuid

import pytest

from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.entity_registry.default import DefaultEntityRegistryService
from memorylayer_server.services.memory.entities import extract_entities as _regex_extract

_DIALOGUE = "[2026-06-01 10:00] Alice: I shipped the Orion release in Paris"


class _SpyTypedProvider:
    """Minimal typed extraction provider used to prove routing.

    Records calls so the test can assert the routing went through the service
    (not the module-level regex util) and returns typed metadata distinct from
    what the regex util would produce.
    """

    def __init__(self):
        self.calls: list[str] = []

    def extract_entities(self, content: str) -> dict:
        self.calls.append(content)
        return {
            "speaker": "Alice",
            "entities": ["Alice", "Acme", "Paris"],
            "entity_types": {
                "Alice": EntityType.PERSON.value,
                "Acme": EntityType.ORG.value,
                "Paris": EntityType.PLACE.value,
            },
        }


def test_maybe_add_entity_metadata_routes_through_service(memory_service):
    """Routing: when a provider is wired, metadata comes from IT, not the regex."""
    spy = _SpyTypedProvider()
    orig_service = memory_service.extraction_service
    orig_anchor = memory_service.entity_anchor_enabled
    memory_service.extraction_service = spy
    memory_service.entity_anchor_enabled = True
    try:
        md = memory_service._maybe_add_entity_metadata(_DIALOGUE, None)
    finally:
        memory_service.extraction_service = orig_service
        memory_service.entity_anchor_enabled = orig_anchor

    # The service WAS called with the content (not bypassed for the regex util).
    assert spy.calls == [_DIALOGUE]
    # Metadata reflects the SPY's typed output, not the regex proper-noun scan.
    assert md["speaker"] == "Alice"
    assert md["entities"] == ["Alice", "Acme", "Paris"]
    assert md["entity_types"] == {
        "Alice": EntityType.PERSON.value,
        "Acme": EntityType.ORG.value,
        "Paris": EntityType.PLACE.value,
    }


def test_maybe_add_entity_metadata_default_provider_unchanged_golden(memory_service):
    """Backward-compat golden: the default regex provider yields metadata
    byte-identical to the standalone regex util (the +16% anchor result must not
    move on the production default)."""
    orig_anchor = memory_service.entity_anchor_enabled
    memory_service.entity_anchor_enabled = True
    try:
        md = memory_service._maybe_add_entity_metadata(_DIALOGUE, None)
    finally:
        memory_service.entity_anchor_enabled = orig_anchor

    golden = _regex_extract(_DIALOGUE)
    assert md["speaker"] == golden["speaker"]
    assert md["entities"] == golden["entities"]
    # Regex provider supplies no types -> entity_types is NOT folded onto md
    # (empty dict is falsy), exactly as before this change.
    assert "entity_types" not in md


def test_maybe_add_entity_metadata_runs_when_only_registry_enabled(memory_service):
    """Gating fix: accretion needs typed metadata even when only the registry
    flag is on (anchor off)."""
    spy = _SpyTypedProvider()
    orig_service = memory_service.extraction_service
    orig_anchor = memory_service.entity_anchor_enabled
    orig_registry = memory_service.entity_registry_enabled
    memory_service.extraction_service = spy
    memory_service.entity_anchor_enabled = False
    memory_service.entity_registry_enabled = True
    try:
        md = memory_service._maybe_add_entity_metadata(_DIALOGUE, None)
    finally:
        memory_service.extraction_service = orig_service
        memory_service.entity_anchor_enabled = orig_anchor
        memory_service.entity_registry_enabled = orig_registry

    assert spy.calls == [_DIALOGUE]
    assert md["entity_types"]["Acme"] == EntityType.ORG.value


def test_maybe_add_entity_metadata_noop_when_both_flags_off(memory_service):
    """No-op when neither anchor nor registry is enabled (metadata unchanged)."""
    spy = _SpyTypedProvider()
    orig_service = memory_service.extraction_service
    orig_anchor = memory_service.entity_anchor_enabled
    orig_registry = memory_service.entity_registry_enabled
    memory_service.extraction_service = spy
    memory_service.entity_anchor_enabled = False
    memory_service.entity_registry_enabled = False
    try:
        md = memory_service._maybe_add_entity_metadata(_DIALOGUE, {"x": 1})
    finally:
        memory_service.extraction_service = orig_service
        memory_service.entity_anchor_enabled = orig_anchor
        memory_service.entity_registry_enabled = orig_registry

    assert spy.calls == []
    assert md == {"x": 1}


async def _new_workspace(storage_backend) -> str:
    ws_id = f"ws-typing-{uuid.uuid4().hex[:8]}"
    await storage_backend.create_workspace(
        Workspace(id=ws_id, tenant_id="_default", name="Typing WS")
    )
    return ws_id


@pytest.mark.asyncio
async def test_accretion_uses_entity_types_from_metadata(memory_service, storage_backend, v):
    """_accrete_entities: entity_types present -> entity gets the mapped type."""
    ws_id = await _new_workspace(storage_backend)
    registry = DefaultEntityRegistryService(storage=storage_backend, v=v)

    spy = _SpyTypedProvider()
    orig_enabled = memory_service.entity_registry_enabled
    orig_service = memory_service.entity_registry_service
    orig_extraction = memory_service.extraction_service
    orig_anchor = memory_service.entity_anchor_enabled
    memory_service.entity_registry_enabled = True
    memory_service.entity_registry_service = registry
    memory_service.extraction_service = spy
    memory_service.entity_anchor_enabled = True
    try:
        await memory_service.remember(ws_id, RememberInput(content=_DIALOGUE), inline=True)
    finally:
        memory_service.entity_registry_enabled = orig_enabled
        memory_service.entity_registry_service = orig_service
        memory_service.extraction_service = orig_extraction
        memory_service.entity_anchor_enabled = orig_anchor

    # Paris was typed PLACE by the provider -> resolves as a PLACE entity.
    place = await registry.resolve(ws_id, "Paris", EntityType.PLACE, allow_create=False)
    assert place.matched_via == "exact"
    # Acme was typed ORG.
    org = await registry.resolve(ws_id, "Acme", EntityType.ORG, allow_create=False)
    assert org.matched_via == "exact"
    # Alice (the speaker) was typed PERSON.
    person = await registry.resolve(ws_id, "Alice", EntityType.PERSON, allow_create=False)
    assert person.matched_via == "exact"


@pytest.mark.asyncio
async def test_accretion_heuristic_when_no_entity_types(memory_service, storage_backend, v):
    """_accrete_entities: entity_types absent -> speaker=PERSON, else CONCEPT."""
    ws_id = await _new_workspace(storage_backend)
    registry = DefaultEntityRegistryService(storage=storage_backend, v=v)

    orig_enabled = memory_service.entity_registry_enabled
    orig_service = memory_service.entity_registry_service
    orig_anchor = memory_service.entity_anchor_enabled
    memory_service.entity_registry_enabled = True
    memory_service.entity_registry_service = registry
    memory_service.entity_anchor_enabled = True  # uses default regex provider (no types)
    try:
        await memory_service.remember(
            ws_id,
            RememberInput(content="[2026-06-01 10:00] Carol: I shipped the Nimbus release"),
            inline=True,
        )
    finally:
        memory_service.entity_registry_enabled = orig_enabled
        memory_service.entity_registry_service = orig_service
        memory_service.entity_anchor_enabled = orig_anchor

    # Speaker -> PERSON heuristic.
    carol = await registry.resolve(ws_id, "Carol", EntityType.PERSON, allow_create=False)
    assert carol.matched_via == "exact"
    # Non-speaker span -> CONCEPT heuristic.
    nimbus = await registry.resolve(ws_id, "Nimbus", EntityType.CONCEPT, allow_create=False)
    assert nimbus.matched_via == "exact"
