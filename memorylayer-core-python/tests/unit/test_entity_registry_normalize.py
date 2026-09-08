"""Unit tests for the entity-name normalization parity contract.

``normalize_entity_name`` is the byte-for-byte key both OSS and enterprise use,
so these tests pin its exact behavior: casefolding, diacritic stripping,
possessive removal, punctuation-to-space, and whitespace collapse.
"""

from memorylayer_server.services.entity_registry import normalize_entity_name


class TestNormalizeEntityName:
    def test_empty_and_whitespace(self):
        assert normalize_entity_name("") == ""
        assert normalize_entity_name("   ") == ""
        assert normalize_entity_name(None) == ""  # type: ignore[arg-type]

    def test_casefold(self):
        assert normalize_entity_name("Alice") == "alice"
        assert normalize_entity_name("ACME Corp") == "acme corp"
        # casefold is more aggressive than lower(): German eszett.
        assert normalize_entity_name("STRASSE") == normalize_entity_name("straße")

    def test_diacritics_stripped(self):
        assert normalize_entity_name("José") == "jose"
        assert normalize_entity_name("Müller") == "muller"
        assert normalize_entity_name("naïve café") == "naive cafe"

    def test_possessive_stripped(self):
        assert normalize_entity_name("Alice's") == "alice"
        assert normalize_entity_name("Alice’s") == "alice"  # typographic apostrophe
        assert normalize_entity_name("James'") == "james"
        # Possessive collapses onto the bare name.
        assert normalize_entity_name("Bob's") == normalize_entity_name("Bob")

    def test_punctuation_to_space(self):
        assert normalize_entity_name("A.C.M.E.") == "a c m e"
        assert normalize_entity_name("Foo-Bar") == "foo bar"
        assert normalize_entity_name("Smith, John") == "smith john"
        assert normalize_entity_name("(OpenAI)") == "openai"

    def test_whitespace_collapsed(self):
        assert normalize_entity_name("  Acme   Corp  ") == "acme corp"
        assert normalize_entity_name("Acme\tCorp\nInc") == "acme corp inc"

    def test_combined(self):
        # All steps together: diacritics + case + possessive + punctuation + ws.
        # Possessive stripping is suffix-only (deterministic, no tokenization),
        # so a mid-string ``'s`` becomes a standalone token rather than vanishing.
        assert normalize_entity_name("  José's Café-Bar  ") == "jose s cafe bar"
        # Trailing possessive is stripped before punctuation handling.
        assert normalize_entity_name("  Café-Bar's  ") == "cafe bar"

    def test_idempotent(self):
        once = normalize_entity_name("José's Café")
        assert normalize_entity_name(once) == once
