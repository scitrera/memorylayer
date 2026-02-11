"""Unit tests for ExtractionService."""
import pytest
from memorylayer_server.services.extraction import ExtractionCategory
from memorylayer_server.services.extraction.default import (
    DefaultExtractionService,
    ExtractionOptions,
    EXTRACTION_SYSTEM_PROMPT,
)


class TestExtractionParsing:
    """Tests for LLM response parsing (no actual LLM needed)."""

    @pytest.fixture
    def extraction_service(self):
        """Create extraction service without LLM for parsing tests."""
        return DefaultExtractionService(
            llm_service=None,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
        )

    def test_parse_valid_json_response(self, extraction_service):
        """Test parsing a valid JSON response."""
        response = '''[
            {
                "content": "User is a Python developer at TechCorp",
                "category": "profile",
                "importance": 0.9,
                "tags": ["developer", "python"]
            },
            {
                "content": "User prefers pytest over unittest",
                "category": "preferences",
                "importance": 0.7,
                "tags": ["testing", "preferences"]
            }
        ]'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 2
        assert result[0].content == "User is a Python developer at TechCorp"
        assert result[0].category == ExtractionCategory.PROFILE
        assert result[0].importance == 0.9
        assert result[0].tags == ["developer", "python"]

        assert result[1].content == "User prefers pytest over unittest"
        assert result[1].category == ExtractionCategory.PREFERENCES
        assert result[1].importance == 0.7

    def test_parse_json_with_markdown_code_block(self, extraction_service):
        """Test parsing JSON wrapped in markdown code block."""
        response = '''```json
[
    {
        "content": "Project Aurora is a microservices migration",
        "category": "entities",
        "importance": 0.8,
        "tags": ["project"]
    }
]
```'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 1
        assert result[0].content == "Project Aurora is a microservices migration"
        assert result[0].category == ExtractionCategory.ENTITIES

    def test_parse_filters_by_category(self, extraction_service):
        """Test that parsing filters by allowed categories."""
        response = '''[
            {"content": "Memory 1", "category": "profile", "importance": 0.8},
            {"content": "Memory 2", "category": "events", "importance": 0.7},
            {"content": "Memory 3", "category": "cases", "importance": 0.9}
        ]'''

        # Only allow profile and cases
        categories = [ExtractionCategory.PROFILE, ExtractionCategory.CASES]
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 2
        assert result[0].category == ExtractionCategory.PROFILE
        assert result[1].category == ExtractionCategory.CASES

    def test_parse_handles_missing_importance(self, extraction_service):
        """Test that missing importance defaults to 0.6."""
        response = '''[
            {"content": "Some memory", "category": "profile"}
        ]'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 1
        assert result[0].importance == 0.6

    def test_parse_clamps_importance(self, extraction_service):
        """Test that importance is clamped to [0, 1]."""
        response = '''[
            {"content": "Memory 1", "category": "profile", "importance": 1.5},
            {"content": "Memory 2", "category": "profile", "importance": -0.5}
        ]'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert result[0].importance == 1.0
        assert result[1].importance == 0.0

    def test_parse_handles_invalid_json(self, extraction_service):
        """Test that invalid JSON returns empty list."""
        response = "This is not JSON"

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert result == []

    def test_parse_handles_non_array_json(self, extraction_service):
        """Test that non-array JSON returns empty list."""
        response = '{"content": "not an array"}'

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert result == []

    def test_parse_skips_invalid_items(self, extraction_service):
        """Test that invalid items are skipped."""
        response = '''[
            {"content": "Valid", "category": "profile", "importance": 0.8},
            {"invalid": "Missing content and category"},
            {"content": "Also valid", "category": "events", "importance": 0.7}
        ]'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 2

    def test_parse_handles_unknown_category(self, extraction_service):
        """Test that unknown categories are skipped."""
        response = '''[
            {"content": "Valid", "category": "profile", "importance": 0.8},
            {"content": "Unknown", "category": "unknown_category", "importance": 0.7}
        ]'''

        categories = list(ExtractionCategory)
        result = extraction_service._parse_llm_response(response, categories)

        assert len(result) == 1
        assert result[0].category == ExtractionCategory.PROFILE


class TestSimpleExtraction:
    """Tests for simple extraction fallback."""

    @pytest.fixture
    def extraction_service(self):
        """Create extraction service without LLM."""
        return DefaultExtractionService(
            llm_service=None,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
        )

    def test_simple_extraction_returns_single_memory(self, extraction_service):
        """Test that simple extraction returns a single memory."""
        context = "User discussed Python development preferences."
        categories = list(ExtractionCategory)
        options = ExtractionOptions()

        result = extraction_service._simple_extraction(context, categories, options)

        assert len(result) == 1
        assert result[0].category == ExtractionCategory.CASES
        assert result[0].importance == 0.6
        assert "auto-extracted" in result[0].tags

    def test_simple_extraction_limits_content_length(self, extraction_service):
        """Test that simple extraction limits content to 1000 chars."""
        context = "x" * 2000
        categories = list(ExtractionCategory)
        options = ExtractionOptions()

        result = extraction_service._simple_extraction(context, categories, options)

        assert len(result[0].content) == 1000

    def test_simple_extraction_handles_empty_context(self, extraction_service):
        """Test that simple extraction handles empty context."""
        context = "   "
        categories = list(ExtractionCategory)
        options = ExtractionOptions()

        result = extraction_service._simple_extraction(context, categories, options)

        assert result == []


class TestExtractionContext:
    """Tests for context building."""

    @pytest.fixture
    def extraction_service(self):
        """Create extraction service."""
        return DefaultExtractionService(
            llm_service=None,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
        )

    def test_build_context_with_session_content_only(self, extraction_service):
        """Test building context with session content only."""
        session_content = "User talked about Python."
        working_memory = {}

        result = extraction_service._build_extraction_context(
            session_content, working_memory
        )

        assert result == "User talked about Python."

    def test_build_context_with_working_memory(self, extraction_service):
        """Test building context with working memory."""
        session_content = "User talked about Python."
        working_memory = {"current_task": "debugging", "framework": "FastAPI"}

        result = extraction_service._build_extraction_context(
            session_content, working_memory
        )

        assert "User talked about Python." in result
        assert "Working Memory:" in result
        assert "current_task: debugging" in result
        assert "framework: FastAPI" in result


class TestExtractionPrompt:
    """Tests for extraction prompt content."""

    def test_system_prompt_contains_all_categories(self):
        """Test that system prompt mentions all categories."""
        assert "profile" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "preferences" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "entities" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "events" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "cases" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "patterns" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_system_prompt_specifies_json_output(self):
        """Test that system prompt specifies JSON output."""
        assert "json" in EXTRACTION_SYSTEM_PROMPT.lower()
        assert "array" in EXTRACTION_SYSTEM_PROMPT.lower()


class TestParsePartialJsonArray:
    """Tests for JSON recovery from malformed LLM output."""

    @pytest.fixture
    def extraction_service(self):
        return DefaultExtractionService(
            llm_service=None,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
        )

    def test_trailing_comma_removed(self, extraction_service):
        """Trailing commas before ] should be cleaned."""
        raw = '[{"content": "fact one"}, {"content": "fact two"},]'
        result = extraction_service._parse_partial_json_array(raw)
        assert len(result) == 2
        assert result[0]["content"] == "fact one"

    def test_truncated_mid_object(self, extraction_service):
        """Truncated JSON should recover complete objects."""
        raw = '[{"content": "fact one", "type": "semantic"}, {"content": "fact tw'
        result = extraction_service._parse_partial_json_array(raw)
        assert len(result) == 1
        assert result[0]["content"] == "fact one"

    def test_truncated_mid_string_value(self, extraction_service):
        """Unterminated string in last object should recover earlier objects."""
        raw = (
            '[{"content": "User prefers Python", "type": "semantic"}, '
            '{"content": "User likes testing with pyt'
        )
        result = extraction_service._parse_partial_json_array(raw)
        assert len(result) == 1
        assert result[0]["content"] == "User prefers Python"

    def test_valid_json_passes_through(self, extraction_service):
        """Valid JSON should parse normally even through recovery path."""
        raw = '[{"content": "fact one"}, {"content": "fact two"}]'
        result = extraction_service._parse_partial_json_array(raw)
        assert len(result) == 2

    def test_completely_unrecoverable_raises(self, extraction_service):
        """Totally invalid input should raise JSONDecodeError."""
        raw = "this is not json at all"
        with pytest.raises(Exception):
            extraction_service._parse_partial_json_array(raw)

    def test_trailing_comma_before_brace(self, extraction_service):
        """Trailing comma before closing brace should be cleaned."""
        raw = '[{"content": "fact one", "type": "semantic",}]'
        result = extraction_service._parse_partial_json_array(raw)
        assert len(result) == 1
        assert result[0]["content"] == "fact one"


class TestCategoryMapping:
    """Tests for category to memory type mapping."""

    def test_all_categories_have_mapping(self):
        """Test that all extraction categories have memory type mapping."""
        from memorylayer_server.services.extraction.base import CATEGORY_MAPPING

        for category in ExtractionCategory:
            assert category in CATEGORY_MAPPING
            memory_type, memory_subtype = CATEGORY_MAPPING[category]
            assert memory_type is not None
