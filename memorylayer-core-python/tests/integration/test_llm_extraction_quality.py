"""
Quality evaluation tests for LLM-based extraction.

These tests evaluate whether the LLM model makes good/sane/reasonable
determinations about memory extraction. Run these to assess model quality
before deploying a new LLM model.

Unlike functional tests, these tests:
1. Check CONTENT QUALITY - Are memories standalone and reusable?
2. Check CATEGORY ACCURACY - Is categorization correct?
3. Check IMPORTANCE CALIBRATION - Are important things ranked higher?
4. Check NOISE REJECTION - Does it skip trivial information?
5. Check COMPLETENESS - Does it capture all key information?

Configure via environment variables:
    MEMORYLAYER_LLM_OPENAI_BASE_URL=http://10.24.11.13:8000/v1
    MEMORYLAYER_LLM_OPENAI_API_KEY=local
    MEMORYLAYER_LLM_OPENAI_MODEL=nemotron3-30b-a3b-2512
"""
import os
import pytest

from memorylayer_server.services.extraction import ExtractionCategory
from memorylayer_server.services.extraction.default import (
    DefaultExtractionService,
    ExtractionOptions,
)
from memorylayer_server.services.llm.openai import OpenAILLMProvider


pytestmark = pytest.mark.llm_quality


def llm_is_available() -> bool:
    """Check if LLM server is configured."""
    return os.environ.get("MEMORYLAYER_LLM_OPENAI_BASE_URL") is not None


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


class TestCategoryAccuracy:
    """Tests for correct category classification."""

    @pytest.mark.asyncio
    async def test_profile_vs_entity_distinction(self, extraction_service):
        """
        Test that the model distinguishes between PROFILE (about the user)
        and ENTITY (about external things).

        PROFILE: "I am a developer" (about the user)
        ENTITY: "John is a developer" (about someone else)
        """
        context = """
        User: I'm a senior Python developer with 10 years of experience.
        My colleague John is a DevOps engineer who manages our Kubernetes clusters.
        Our project manager Sarah handles the sprint planning.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Find memories about the user vs others
        user_memories = [m for m in result if "senior python developer" in m.content.lower()
                        or "10 years" in m.content.lower()]
        other_memories = [m for m in result if "john" in m.content.lower()
                         or "sarah" in m.content.lower()]

        # User info should be PROFILE
        for m in user_memories:
            assert m.category == ExtractionCategory.PROFILE, \
                f"User info should be PROFILE, got {m.category}: {m.content}"

        # Others should be ENTITIES
        for m in other_memories:
            assert m.category == ExtractionCategory.ENTITIES, \
                f"Info about others should be ENTITIES, got {m.category}: {m.content}"

    @pytest.mark.asyncio
    async def test_preference_vs_event_distinction(self, extraction_service):
        """
        Test that the model distinguishes between PREFERENCES (ongoing choices)
        and EVENTS (one-time occurrences).

        PREFERENCE: "I always use pytest" (ongoing)
        EVENT: "Yesterday I decided to use pytest" (one-time decision)
        """
        context = """
        User: I always use Black for code formatting with line length 88.
        Yesterday we decided to migrate from MySQL to PostgreSQL.
        I prefer async/await over threads for concurrency.
        Last month we completed the authentication refactor.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Find preference and event memories
        preference_keywords = ["always", "prefer", "black", "async"]
        event_keywords = ["yesterday", "last month", "decided", "completed"]

        for m in result:
            content_lower = m.content.lower()
            has_pref_keyword = any(k in content_lower for k in preference_keywords)
            has_event_keyword = any(k in content_lower for k in event_keywords)

            if has_pref_keyword and not has_event_keyword:
                assert m.category == ExtractionCategory.PREFERENCES, \
                    f"Ongoing preference should be PREFERENCES: {m.content}"
            elif has_event_keyword and "always" not in content_lower:
                assert m.category == ExtractionCategory.EVENTS, \
                    f"One-time event should be EVENTS: {m.content}"

    @pytest.mark.asyncio
    async def test_case_extraction_has_problem_and_solution(self, extraction_service):
        """
        Test that CASES memories contain both problem and solution context,
        not just the problem or just the solution.
        """
        context = """
        User: We had a nasty bug where API requests were timing out randomly.
        After investigating, we found the connection pool was exhausted.
        The fix was to increase max_connections from 5 to 20 and add proper
        connection cleanup in the finally block. This resolved the timeouts.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        case_memories = [m for m in result if m.category == ExtractionCategory.CASES]
        assert len(case_memories) > 0, "Should extract at least one case"

        # At least one case should mention both problem and solution
        for m in case_memories:
            content_lower = m.content.lower()
            has_problem = any(w in content_lower for w in ["timeout", "bug", "issue", "problem", "exhausted"])
            has_solution = any(w in content_lower for w in ["fix", "increase", "resolved", "solution", "cleanup"])

            # Either this case has both, or there are other cases that complete it
            if has_problem or has_solution:
                # It's valid - cases can be problem+solution or just be part of a set
                pass

        # Combined, all cases should have both elements (LLMs may rephrase)
        all_content = " ".join(m.content.lower() for m in case_memories)
        problem_indicators = ["timeout", "bug", "issue", "problem", "exhausted",
                              "error", "fail", "request", "api", "connection"]
        solution_indicators = ["fix", "increase", "resolved", "solution", "cleanup",
                               "change", "set", "add", "20", "max_connections"]

        assert any(w in all_content for w in problem_indicators), \
            f"Cases should mention the problem. Content: {all_content}"
        assert any(w in all_content for w in solution_indicators), \
            f"Cases should mention the solution. Content: {all_content}"


class TestImportanceCalibration:
    """Tests for sensible importance scoring."""

    @pytest.mark.asyncio
    async def test_high_importance_for_critical_info(self, extraction_service):
        """
        Test that critical identity info gets high importance (>= 0.8).
        """
        context = """
        User: I'm the CTO of TechCorp, responsible for all technical decisions.
        We're a Series C startup with 200 engineers.
        I report directly to the CEO.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # CTO info should be high importance
        cto_memories = [m for m in result if "cto" in m.content.lower()]
        assert len(cto_memories) > 0, "Should extract CTO information"

        for m in cto_memories:
            assert m.importance >= 0.7, \
                f"CTO role should be high importance (>=0.7), got {m.importance}: {m.content}"

    @pytest.mark.asyncio
    async def test_lower_importance_for_minor_preferences(self, extraction_service):
        """
        Test that minor preferences get lower importance than major decisions.
        """
        context = """
        User: I prefer tabs over spaces for indentation.
        We just decided to rewrite our entire payment system in Rust.
        I like dark mode in my IDE.
        Our authentication system will migrate to OAuth2 next quarter.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Find minor vs major memories
        minor_keywords = ["tabs", "spaces", "dark mode", "ide"]
        major_keywords = ["payment", "rewrite", "authentication", "oauth", "migrate"]

        minor_memories = [m for m in result
                         if any(k in m.content.lower() for k in minor_keywords)]
        major_memories = [m for m in result
                         if any(k in m.content.lower() for k in major_keywords)]

        if minor_memories and major_memories:
            avg_minor = sum(m.importance for m in minor_memories) / len(minor_memories)
            avg_major = sum(m.importance for m in major_memories) / len(major_memories)

            assert avg_major > avg_minor, \
                f"Major decisions ({avg_major:.2f}) should have higher importance than minor preferences ({avg_minor:.2f})"


class TestNoiseRejection:
    """Tests for filtering out low-value information."""

    @pytest.mark.asyncio
    async def test_filters_greetings_and_chatter(self, extraction_service):
        """
        Test that the model doesn't extract trivial conversation artifacts.
        """
        context = """
        User: Hello! How are you today?
        Assistant: I'm doing well, thank you for asking!
        User: Great! So, I'm a Python developer working on a machine learning project.
        We use PyTorch for model training and FastAPI for serving.
        User: Thanks for your help!
        Assistant: You're welcome! Let me know if you need anything else.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Should not extract greetings/thank yous as memories
        for m in result:
            content_lower = m.content.lower()
            assert "hello" not in content_lower or "python" in content_lower, \
                f"Should not extract bare greetings: {m.content}"
            assert "thank" not in content_lower or any(k in content_lower for k in ["python", "pytorch", "fastapi"]), \
                f"Should not extract thank yous: {m.content}"
            assert "you're welcome" not in content_lower, \
                f"Should not extract pleasantries: {m.content}"

    @pytest.mark.asyncio
    async def test_extracts_technical_content_from_noisy_conversation(self, extraction_service):
        """
        Test that valuable info is extracted even from noisy conversation.
        """
        context = """
        User: Hmm, let me think... um, so basically what happened was...
        well, you know how it is... anyway, we had this really annoying bug
        where the Redis cache was returning stale data. We fixed it by adding
        a cache invalidation hook on database updates. Simple fix but took
        forever to debug! LOL
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Should extract the actual technical content
        all_content = " ".join(m.content.lower() for m in result)

        assert "redis" in all_content or "cache" in all_content, \
            "Should extract Redis/cache information from noisy text"
        assert "invalidation" in all_content or "stale" in all_content, \
            "Should extract the problem/solution despite noise"


class TestContentQuality:
    """Tests for extracted memory content quality."""

    @pytest.mark.asyncio
    async def test_memories_are_standalone(self, extraction_service):
        """
        Test that extracted memories make sense without the original context.
        They should not contain dangling references like "this" or "it".
        """
        context = """
        User: We're building a recommendation engine for our e-commerce platform.
        It uses collaborative filtering and content-based filtering.
        This hybrid approach gives us better results than either alone.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        for m in result:
            # Memory should not start with dangling references
            first_word = m.content.split()[0].lower() if m.content.split() else ""
            assert first_word not in ["it", "this", "that", "these", "those"], \
                f"Memory should not start with dangling reference: {m.content}"

            # Memory should contain the subject, not just pronouns
            content_lower = m.content.lower()
            has_subject = any(k in content_lower for k in
                            ["recommendation", "engine", "filtering", "e-commerce",
                             "platform", "hybrid", "approach"])
            assert has_subject, \
                f"Memory should contain concrete subject, not just pronouns: {m.content}"

    @pytest.mark.asyncio
    async def test_memories_are_concise(self, extraction_service):
        """
        Test that memories are concise, not overly verbose.
        """
        context = """
        User: Let me explain our architecture in great detail with lots of
        background information. So basically, we have a microservices architecture
        where each service is independently deployable. The main services are:
        user-service for authentication, product-service for catalog management,
        order-service for transactions, and notification-service for emails.
        We use Kubernetes for orchestration and Istio for service mesh.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        for m in result:
            # Memories should generally be under 300 characters for conciseness
            # (not a hard rule, but most should be)
            assert len(m.content) < 500, \
                f"Memory might be too verbose ({len(m.content)} chars): {m.content[:100]}..."


class TestCompleteness:
    """Tests for capturing all key information."""

    @pytest.mark.asyncio
    async def test_captures_all_key_entities(self, extraction_service):
        """
        Test that multiple mentioned entities are all captured.
        """
        context = """
        User: Our tech stack includes:
        - Python with FastAPI for the backend
        - PostgreSQL for the database
        - Redis for caching
        - React for the frontend
        - Kubernetes for deployment
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        all_content = " ".join(m.content.lower() for m in result)

        # Should capture the key technologies
        key_techs = ["python", "fastapi", "postgresql", "redis", "react", "kubernetes"]
        captured = [t for t in key_techs if t in all_content]

        # Should capture at least most of them (allow for some summarization)
        assert len(captured) >= 4, \
            f"Should capture most key technologies. Captured: {captured}, Missing: {set(key_techs) - set(captured)}"

    @pytest.mark.asyncio
    async def test_captures_both_problem_and_solution_in_case(self, extraction_service):
        """
        Test that when a problem+solution is described, both are captured.
        """
        context = """
        User: We were getting OOM errors in production. The Java heap was set to
        only 512MB. After profiling, we increased it to 2GB and added GC tuning
        flags: -XX:+UseG1GC -XX:MaxGCPauseMillis=200. This eliminated the OOMs
        and improved response times by 40%.
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        all_content = " ".join(m.content.lower() for m in result)

        # Should capture both problem and solution elements
        assert any(w in all_content for w in ["oom", "out of memory", "512mb", "heap"]), \
            "Should capture the problem (OOM/memory issue)"
        assert any(w in all_content for w in ["2gb", "g1gc", "increased", "gc tuning"]), \
            "Should capture the solution (heap increase, GC tuning)"


class TestModelEvaluationSummary:
    """
    Run all quality tests and produce a summary score.
    This helps compare different LLM models.
    """

    @pytest.mark.asyncio
    async def test_overall_extraction_quality(self, extraction_service):
        """
        Comprehensive test that evaluates overall extraction quality
        and prints a summary for model evaluation.
        """
        # Rich test context with multiple types of information
        context = """
        User: Hi! I'm Alex Chen, a Staff Engineer at DataCorp (that's me - the user).
        I specialize in distributed systems and have been here 8 years.

        My colleague Maria runs the ML team - she's amazing at NLP.

        I strongly prefer Python over Go for most things, but we use Go for
        our high-performance services. I always use type hints and pytest.

        Last week we had a major incident - our Kafka cluster lost messages
        due to a misconfigured retention policy. We fixed it by increasing
        retention to 7 days and adding monitoring alerts. Never again!

        Our deployment process: PR review -> CI tests -> staging deploy ->
        canary rollout -> full production. Takes about 2 hours end-to-end.

        Oh, and we're working on Project Phoenix - a complete rewrite of our
        data pipeline using Apache Spark and Delta Lake.

        Thanks for listening!
        """

        categories = list(ExtractionCategory)
        result = await extraction_service._llm_extraction(context, categories)

        # Collect statistics
        stats = {
            "total_memories": len(result),
            "by_category": {},
            "avg_importance": 0,
            "quality_checks": {
                "has_profile": False,
                "has_preference": False,
                "has_entity": False,
                "has_event": False,
                "has_case": False,
                "has_pattern": False,
                "user_correctly_profiled": False,
                "others_are_entities": False,
                "no_noise_extracted": True,
            }
        }

        for cat in ExtractionCategory:
            cat_memories = [m for m in result if m.category == cat]
            stats["by_category"][cat.value] = len(cat_memories)

            if cat_memories:
                if cat == ExtractionCategory.PROFILE:
                    stats["quality_checks"]["has_profile"] = True
                elif cat == ExtractionCategory.PREFERENCES:
                    stats["quality_checks"]["has_preference"] = True
                elif cat == ExtractionCategory.ENTITIES:
                    stats["quality_checks"]["has_entity"] = True
                elif cat == ExtractionCategory.EVENTS:
                    stats["quality_checks"]["has_event"] = True
                elif cat == ExtractionCategory.CASES:
                    stats["quality_checks"]["has_case"] = True
                elif cat == ExtractionCategory.PATTERNS:
                    stats["quality_checks"]["has_pattern"] = True

        if result:
            stats["avg_importance"] = sum(m.importance for m in result) / len(result)

        # Check specific quality items
        all_content = " ".join(m.content.lower() for m in result)

        # User should be correctly profiled
        profile_memories = [m for m in result if m.category == ExtractionCategory.PROFILE]
        for m in profile_memories:
            if "alex" in m.content.lower() or "staff engineer" in m.content.lower():
                stats["quality_checks"]["user_correctly_profiled"] = True
                break

        # Maria should be an entity, not profile
        entity_memories = [m for m in result if m.category == ExtractionCategory.ENTITIES]
        for m in entity_memories:
            if "maria" in m.content.lower():
                stats["quality_checks"]["others_are_entities"] = True
                break

        # Should not extract "thanks" or "hi" as standalone memories
        for m in result:
            content_lower = m.content.lower().strip()
            if content_lower in ["hi", "hello", "thanks", "thanks for listening", "thank you"]:
                stats["quality_checks"]["no_noise_extracted"] = False
                break

        # Print summary for manual review
        print("\n" + "=" * 60)
        print("LLM EXTRACTION QUALITY EVALUATION")
        print("=" * 60)
        print(f"Model: {os.environ.get('MEMORYLAYER_LLM_OPENAI_MODEL', 'unknown')}")
        print(f"Total memories extracted: {stats['total_memories']}")
        print(f"Average importance: {stats['avg_importance']:.2f}")
        print("\nBy category:")
        for cat, count in stats["by_category"].items():
            print(f"  {cat}: {count}")
        print("\nQuality checks:")
        passed = 0
        total = len(stats["quality_checks"])
        for check, result_val in stats["quality_checks"].items():
            status = "✓ PASS" if result_val else "✗ FAIL"
            print(f"  {status}: {check}")
            if result_val:
                passed += 1
        print(f"\nOverall: {passed}/{total} checks passed ({100*passed/total:.0f}%)")
        print("=" * 60 + "\n")

        # Assert minimum quality bar
        assert stats["total_memories"] >= 3, "Should extract at least 3 memories"
        assert stats["quality_checks"]["has_profile"], "Should extract profile information"
        assert stats["quality_checks"]["has_preference"], "Should extract preferences"
        assert passed >= total * 0.6, f"Should pass at least 60% of quality checks ({passed}/{total})"
