"""
Integration tests for LLM-based extraction.

These tests require a running LLM server (e.g., vLLM, OpenAI-compatible endpoint).
Skip with: pytest -m "not llm" to skip when LLM is not available.

Configure via environment variables:
    MEMORYLAYER_LLM_OPENAI_BASE_URL=http://10.24.11.13:8000/v1
    MEMORYLAYER_LLM_OPENAI_API_KEY=local
    MEMORYLAYER_LLM_OPENAI_MODEL=nemotron3-30b-a3b-2512
"""
import os
import pytest
import pytest_asyncio

from memorylayer_server.services.extraction import ExtractionCategory
from memorylayer_server.services.extraction.default import (
    DefaultExtractionService,
    ExtractionOptions,
)
from memorylayer_server.services.llm.openai import OpenAILLMProvider


# Skip all tests in this module if LLM is not configured
pytestmark = pytest.mark.llm


def llm_is_available() -> bool:
    """Check if LLM server is configured and available."""
    base_url = os.environ.get("MEMORYLAYER_LLM_OPENAI_BASE_URL")
    return base_url is not None


@pytest.fixture
def llm_provider():
    """Create OpenAI LLM provider from environment."""
    if not llm_is_available():
        pytest.skip("LLM server not configured")

    return OpenAILLMProvider(
        api_key=os.environ.get("MEMORYLAYER_LLM_OPENAI_API_KEY", "local"),
        base_url=os.environ.get("MEMORYLAYER_LLM_OPENAI_BASE_URL"),
        model=os.environ.get("MEMORYLAYER_LLM_OPENAI_MODEL", "gpt-4o-mini"),
    )


@pytest.fixture
def extraction_service(llm_provider):
    """Create extraction service with LLM provider."""
    return DefaultExtractionService(
        llm_provider=llm_provider,
        storage=None,
        deduplication_service=None,
        embedding_service=None,
    )


class TestLLMExtraction:
    """Integration tests for LLM-based extraction."""

    @pytest.mark.asyncio
    async def test_extract_profile_memory(self, extraction_service):
        """Test extracting profile information."""
        context = """
        User: Hi, I'm Sarah Chen. I'm a senior software engineer at TechCorp,
        specializing in distributed systems and Python development.
        I've been working here for 5 years and lead the platform team.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        assert len(result) > 0
        # Should extract at least one profile memory
        profile_memories = [m for m in result if m.category == ExtractionCategory.PROFILE]
        assert len(profile_memories) > 0, "Should extract profile information"

        # Verify content quality
        profile_content = " ".join(m.content for m in profile_memories)
        assert any(term in profile_content.lower() for term in ["sarah", "engineer", "techcorp"])

    @pytest.mark.asyncio
    async def test_extract_preference_memory(self, extraction_service):
        """Test extracting user preferences."""
        context = """
        User: I always use pytest for testing - never unittest.
        I prefer FastAPI over Flask for API development.
        For formatting, I use Black with a line length of 88.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        assert len(result) > 0
        # Should extract preferences
        pref_memories = [m for m in result if m.category == ExtractionCategory.PREFERENCES]
        assert len(pref_memories) > 0, "Should extract preferences"

    @pytest.mark.asyncio
    async def test_extract_case_memory(self, extraction_service):
        """Test extracting problem/solution cases."""
        context = """
        User: We had a memory leak in production. After investigating,
        we found that the database connections weren't being properly closed.
        The fix was to add a connection pool with max_connections=20 and
        ensure all connections are returned to the pool after use.
        This reduced memory usage by 60%.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        assert len(result) > 0
        # Should extract case/solution
        case_memories = [m for m in result if m.category == ExtractionCategory.CASES]
        assert len(case_memories) > 0, "Should extract problem/solution case"

    @pytest.mark.asyncio
    async def test_extract_pattern_memory(self, extraction_service):
        """Test extracting workflow patterns."""
        context = """
        User: Our deployment process is:
        1. Run all tests with pytest
        2. Build the Docker image with docker build -t app:latest .
        3. Push to our private registry
        4. Update the Kubernetes deployment
        5. Wait for rollout to complete
        6. Run smoke tests against the new deployment
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        assert len(result) > 0
        # Should extract workflow pattern
        pattern_memories = [m for m in result if m.category == ExtractionCategory.PATTERNS]
        assert len(pattern_memories) > 0, "Should extract workflow pattern"

    @pytest.mark.asyncio
    async def test_extract_multiple_categories(self, extraction_service):
        """Test extracting memories from mixed content."""
        context = """
        User: I'm Alex from DataCorp (profile). We're working on Project Mercury,
        a real-time analytics platform (entity). I prefer using PostgreSQL for
        this kind of workload (preference). Last week we decided to switch from
        Kafka to Redpanda for message streaming (event). When we had issues with
        message ordering, we solved it by using partition keys based on customer ID (case).
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Should extract multiple categories
        categories_found = set(m.category for m in result)
        assert len(categories_found) >= 3, f"Should extract multiple categories, got: {categories_found}"

    @pytest.mark.asyncio
    async def test_extract_with_empty_content(self, extraction_service):
        """Test extraction with minimal content."""
        context = "Hello"

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Should return empty or minimal results for trivial content
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_importance_scoring(self, extraction_service):
        """Test that importance scores are reasonable."""
        context = """
        User: I'm the CTO of MegaCorp, responsible for all technical decisions.
        I like using tabs over spaces for indentation.
        We just closed a $50M funding round.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        assert len(result) > 0
        # All importance scores should be valid
        for memory in result:
            assert 0.0 <= memory.importance <= 1.0

    @pytest.mark.asyncio
    async def test_full_extraction_flow(self, extraction_service):
        """Test the complete extract_from_session flow."""
        session_content = """
        User: Let me tell you about our project. I'm working on MemoryLayer,
        an AI memory system. We use Python with FastAPI for the backend.

        Today we decided to implement vector search using pgvector.
        The key insight was that we needed tiered storage - hot memories in
        Redis, warm in PostgreSQL, cold in S3.

        Our workflow is: write tests first, implement, review, deploy.
        """

        options = ExtractionOptions(
            min_importance=0.5,
            deduplicate=False,  # Skip deduplication for this test
            max_memories=10,
        )

        result = await extraction_service.extract_from_session(
            session_id="test_session",
            workspace_id="test_workspace",
            context_id="test_context",
            session_content=session_content,
            working_memory={},
            options=options,
        )

        assert result.session_id == "test_session"
        assert result.memories_extracted > 0
        assert len(result.memories_created) > 0
        assert result.extraction_time_ms >= 0

        # Check breakdown contains categories
        assert len(result.breakdown) > 0
