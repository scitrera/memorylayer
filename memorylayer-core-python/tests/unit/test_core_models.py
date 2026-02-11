"""
Comprehensive unit tests for core domain models.

Tests all domain models, enums, validators, and factory methods
from the MemoryLayer.ai core modules.
"""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from memorylayer_server.models.association import (
    Association,
    AssociateInput,
    GraphPath,
    GraphQueryInput,
    GraphQueryResult,
    RelationshipCategory,
    KNOWN_RELATIONSHIP_TYPES,
    get_relationship_category,
)
from memorylayer_server.models.memory import (
    Memory,
    MemorySubtype,
    MemoryType,
    RecallInput,
    RecallMode,
    RecallResult,
    ReflectInput,
    ReflectResult,
    RememberInput,
    SearchTolerance,
)
from memorylayer_server.models.session import (
    ActivitySummary,
    Contradiction,
    OpenThread,
    Session,
    SessionBriefing,
    WorkingMemory,
    WorkspaceSummary,
)
from memorylayer_server.models.workspace import (
    Context,
    ContextSettings,
    Workspace,
    WorkspaceSettings,
)


# ============================================================================
# MEMORY MODEL TESTS
# ============================================================================


class TestMemoryEnums:
    """Test Memory-related enumerations."""

    def test_memory_type_enum_values(self):
        """Test all MemoryType enum values."""
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.WORKING.value == "working"

        # Verify all expected values exist
        assert set(MemoryType) == {
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
            MemoryType.WORKING,
        }

    def test_memory_subtype_enum_values(self):
        """Test all MemorySubtype enum values."""
        assert MemorySubtype.SOLUTION.value == "solution"
        assert MemorySubtype.PROBLEM.value == "problem"
        assert MemorySubtype.CODE_PATTERN.value == "code_pattern"
        assert MemorySubtype.FIX.value == "fix"
        assert MemorySubtype.ERROR.value == "error"
        assert MemorySubtype.WORKFLOW.value == "workflow"
        assert MemorySubtype.PREFERENCE.value == "preference"
        assert MemorySubtype.DECISION.value == "decision"
        assert MemorySubtype.PROFILE.value == "profile"
        assert MemorySubtype.ENTITY.value == "entity"
        assert MemorySubtype.EVENT.value == "event"
        assert MemorySubtype.DIRECTIVE.value == "directive"

        # Verify all expected values exist
        assert set(MemorySubtype) == {
            MemorySubtype.SOLUTION,
            MemorySubtype.PROBLEM,
            MemorySubtype.CODE_PATTERN,
            MemorySubtype.FIX,
            MemorySubtype.ERROR,
            MemorySubtype.WORKFLOW,
            MemorySubtype.PREFERENCE,
            MemorySubtype.DECISION,
            MemorySubtype.PROFILE,
            MemorySubtype.ENTITY,
            MemorySubtype.EVENT,
            MemorySubtype.DIRECTIVE,
        }

    def test_recall_mode_enum_values(self):
        """Test all RecallMode enum values."""
        assert RecallMode.RAG.value == "rag"
        assert RecallMode.LLM.value == "llm"
        assert RecallMode.HYBRID.value == "hybrid"

        assert set(RecallMode) == {RecallMode.RAG, RecallMode.LLM, RecallMode.HYBRID}

    def test_search_tolerance_enum_values(self):
        """Test all SearchTolerance enum values."""
        assert SearchTolerance.LOOSE.value == "loose"
        assert SearchTolerance.MODERATE.value == "moderate"
        assert SearchTolerance.STRICT.value == "strict"

        assert set(SearchTolerance) == {
            SearchTolerance.LOOSE,
            SearchTolerance.MODERATE,
            SearchTolerance.STRICT,
        }


class TestMemoryModel:
    """Test Memory model validation."""

    def test_memory_content_not_empty(self):
        """Test that memory content cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            Memory(
                id="mem-1",
                workspace_id="ws-1",
                tenant_id="default_tenant",
                content="",
                content_hash="hash",
                type=MemoryType.EPISODIC,
            )

        errors = exc_info.value.errors()
        assert any("content" in str(e.get("loc")) for e in errors)

    def test_memory_content_whitespace_stripped(self):
        """Test that content whitespace is stripped."""
        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="  valid content  ",
            content_hash="hash",
            type=MemoryType.EPISODIC,
        )
        assert memory.content == "valid content"

    def test_memory_importance_bounds(self):
        """Test importance must be between 0 and 1."""
        # Valid bounds
        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="test",
            content_hash="hash",
            type=MemoryType.EPISODIC,
            importance=0.0,
        )
        assert memory.importance == 0.0

        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="test",
            content_hash="hash",
            type=MemoryType.EPISODIC,
            importance=1.0,
        )
        assert memory.importance == 1.0

        # Invalid - below bounds
        with pytest.raises(ValidationError):
            Memory(
                id="mem-1",
                workspace_id="ws-1",
                tenant_id="default_tenant",
                content="test",
                content_hash="hash",
                type=MemoryType.EPISODIC,
                importance=-0.1,
            )

        # Invalid - above bounds
        with pytest.raises(ValidationError):
            Memory(
                id="mem-1",
                workspace_id="ws-1",
                tenant_id="default_tenant",
                content="test",
                content_hash="hash",
                type=MemoryType.EPISODIC,
                importance=1.1,
            )

    def test_memory_tags_normalization(self):
        """Test that tags are normalized (lowercase, deduplicated, sorted)."""
        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="test",
            content_hash="hash",
            type=MemoryType.EPISODIC,
            tags=["Python", "TESTING", "python", "  testing  ", "api"],
        )

        # Should be lowercase, deduplicated, sorted
        assert memory.tags == ["api", "python", "testing"]

    def test_memory_tags_empty_filtered(self):
        """Test that empty tags are filtered out."""
        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="test",
            content_hash="hash",
            type=MemoryType.EPISODIC,
            tags=["valid", "", "  ", "another"],
        )

        assert memory.tags == ["another", "valid"]

    def test_memory_default_values(self):
        """Test Memory model default values."""
        memory = Memory(
            id="mem-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            content="test",
            content_hash="hash",
            type=MemoryType.EPISODIC,
        )

        assert memory.importance == 0.5
        assert memory.tags == []
        assert memory.metadata == {}
        assert memory.access_count == 0
        assert memory.decay_factor == 1.0
        assert memory.last_accessed_at is None
        assert memory.embedding is None


class TestRememberInput:
    """Test RememberInput validation."""

    def test_remember_input_content_required(self):
        """Test that content is required."""
        with pytest.raises(ValidationError):
            RememberInput()

    def test_remember_input_importance_bounds(self):
        """Test importance validation."""
        # Valid
        input_data = RememberInput(content="test", importance=0.5)
        assert input_data.importance == 0.5

        # Invalid - below bounds
        with pytest.raises(ValidationError):
            RememberInput(content="test", importance=-0.1)

        # Invalid - above bounds
        with pytest.raises(ValidationError):
            RememberInput(content="test", importance=1.5)

    def test_remember_input_defaults(self):
        """Test RememberInput default values."""
        input_data = RememberInput(content="test")

        assert input_data.importance == 0.5
        assert input_data.tags == []
        assert input_data.metadata == {}
        assert input_data.associations == []


class TestRecallInput:
    """Test RecallInput validation."""

    def test_recall_input_query_required(self):
        """Test that query is required."""
        with pytest.raises(ValidationError):
            RecallInput()

    def test_recall_input_limit_bounds(self):
        """Test limit must be between 1 and 100."""
        # Valid bounds
        input_data = RecallInput(query="test", limit=1)
        assert input_data.limit == 1

        input_data = RecallInput(query="test", limit=100)
        assert input_data.limit == 100

        # Invalid - below bounds
        with pytest.raises(ValidationError):
            RecallInput(query="test", limit=0)

        # Invalid - above bounds
        with pytest.raises(ValidationError):
            RecallInput(query="test", limit=101)

    def test_recall_input_min_relevance_bounds(self):
        """Test min_relevance must be between 0 and 1."""
        # Valid
        input_data = RecallInput(query="test", min_relevance=0.0)
        assert input_data.min_relevance == 0.0

        input_data = RecallInput(query="test", min_relevance=1.0)
        assert input_data.min_relevance == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            RecallInput(query="test", min_relevance=-0.1)

        with pytest.raises(ValidationError):
            RecallInput(query="test", min_relevance=1.1)

    def test_recall_input_defaults(self):
        """Test RecallInput default values."""
        input_data = RecallInput(query="test query")

        assert input_data.mode is None
        assert input_data.tolerance is None
        assert input_data.detail_level is None
        assert input_data.limit == 10
        assert input_data.min_relevance is None
        assert input_data.include_associations is None
        assert input_data.traverse_depth is None
        assert input_data.max_expansion is None


class TestReflectInput:
    """Test ReflectInput validation."""

    @pytest.mark.skip(reason="Out of date token bounds; need to update test")
    def test_reflect_input_max_tokens_bounds(self):
        """Test max_tokens must be between 50 and 4000."""
        # Valid bounds
        input_data = ReflectInput(query="test", max_tokens=50)
        assert input_data.max_tokens == 50

        input_data = ReflectInput(query="test", max_tokens=4000)
        assert input_data.max_tokens == 4000

        # Invalid - below bounds
        with pytest.raises(ValidationError):
            ReflectInput(query="test", max_tokens=49)

        # Invalid - above bounds
        with pytest.raises(ValidationError):
            ReflectInput(query="test", max_tokens=4001)

    @pytest.mark.skip(reason="Out of date input defaults; need to update test")
    def test_reflect_input_defaults(self):
        """Test ReflectInput default values."""
        input_data = ReflectInput(query="test query")

        assert input_data.max_tokens == 500
        assert input_data.include_sources is True
        assert input_data.depth == 2


# ============================================================================
# ASSOCIATION MODEL TESTS
# ============================================================================


class TestRelationshipEnums:
    """Test Association-related enumerations and string-based relationship types."""

    def test_relationship_category_enum(self):
        """Test all RelationshipCategory enum values."""
        assert RelationshipCategory.HIERARCHICAL.value == "hierarchical"
        assert RelationshipCategory.CAUSAL.value == "causal"
        assert RelationshipCategory.TEMPORAL.value == "temporal"
        assert RelationshipCategory.SIMILARITY.value == "similarity"
        assert RelationshipCategory.LEARNING.value == "learning"
        assert RelationshipCategory.REFINEMENT.value == "refinement"
        assert RelationshipCategory.REFERENCE.value == "reference"
        assert RelationshipCategory.SOLUTION.value == "solution"
        assert RelationshipCategory.CONTEXT.value == "context"
        assert RelationshipCategory.WORKFLOW.value == "workflow"
        assert RelationshipCategory.QUALITY.value == "quality"

        assert set(RelationshipCategory) == {
            RelationshipCategory.HIERARCHICAL,
            RelationshipCategory.CAUSAL,
            RelationshipCategory.TEMPORAL,
            RelationshipCategory.SIMILARITY,
            RelationshipCategory.LEARNING,
            RelationshipCategory.REFINEMENT,
            RelationshipCategory.REFERENCE,
            RelationshipCategory.SOLUTION,
            RelationshipCategory.CONTEXT,
            RelationshipCategory.WORKFLOW,
            RelationshipCategory.QUALITY,
        }

    def test_known_relationship_types_contains_core_types(self):
        """Test KNOWN_RELATIONSHIP_TYPES contains all expected core types."""
        # Causal
        for rel in ("causes", "triggers", "leads_to", "prevents"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Solution
        for rel in ("solves", "addresses", "alternative_to", "improves"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Context
        for rel in ("occurs_in", "applies_to", "works_with", "requires"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Learning
        for rel in ("builds_on", "contradicts", "confirms", "supersedes"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Similarity
        for rel in ("similar_to", "variant_of", "related_to"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Workflow
        for rel in ("follows", "depends_on", "enables", "blocks"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

        # Quality
        for rel in ("effective_for", "preferred_over", "deprecated_by"):
            assert rel in KNOWN_RELATIONSHIP_TYPES, f"{rel} missing from KNOWN_RELATIONSHIP_TYPES"

    def test_known_relationship_types_are_lowercase_strings(self):
        """Test all KNOWN_RELATIONSHIP_TYPES are lowercase strings."""
        for rel in KNOWN_RELATIONSHIP_TYPES:
            assert isinstance(rel, str), f"{rel} is not a string"
            assert rel == rel.lower(), f"{rel} is not lowercase"

    def test_get_relationship_category_causal(self):
        """Test get_relationship_category for causal relationships."""
        assert get_relationship_category("causes") == "causal"
        assert get_relationship_category("triggers") == "causal"
        assert get_relationship_category("leads_to") == "causal"
        assert get_relationship_category("prevents") == "causal"
        assert get_relationship_category("enables") == "causal"

    def test_get_relationship_category_solution(self):
        """Test get_relationship_category for solution relationships."""
        assert get_relationship_category("solves") == "solution"
        assert get_relationship_category("addresses") == "solution"
        assert get_relationship_category("alternative_to") == "solution"
        assert get_relationship_category("improves") == "solution"

    def test_get_relationship_category_context(self):
        """Test get_relationship_category for context relationships."""
        assert get_relationship_category("occurs_in") == "context"
        assert get_relationship_category("applies_to") == "context"
        assert get_relationship_category("works_with") == "context"
        assert get_relationship_category("requires") == "context"

    def test_get_relationship_category_learning(self):
        """Test get_relationship_category for learning relationships."""
        assert get_relationship_category("builds_on") == "learning"
        assert get_relationship_category("contradicts") == "learning"
        assert get_relationship_category("confirms") == "learning"
        assert get_relationship_category("supersedes") == "learning"

    def test_get_relationship_category_similarity(self):
        """Test get_relationship_category for similarity relationships."""
        assert get_relationship_category("similar_to") == "similarity"
        assert get_relationship_category("variant_of") == "similarity"
        assert get_relationship_category("related_to") == "similarity"

    def test_get_relationship_category_workflow(self):
        """Test get_relationship_category for workflow relationships."""
        assert get_relationship_category("follows") == "workflow"
        assert get_relationship_category("depends_on") == "workflow"
        assert get_relationship_category("blocks") == "workflow"
        # Note: enables is categorized as 'causal' in the unified ontology

    def test_get_relationship_category_quality(self):
        """Test get_relationship_category for quality relationships."""
        assert get_relationship_category("effective_for") == "quality"
        assert get_relationship_category("preferred_over") == "quality"
        assert get_relationship_category("deprecated_by") == "quality"

    def test_get_relationship_category_unknown(self):
        """Test get_relationship_category returns None for unknown types."""
        assert get_relationship_category("UNKNOWN_TYPE") is None
        assert get_relationship_category("") is None


class TestAssociationModel:
    """Test Association model validation."""

    def test_association_strength_bounds(self):
        """Test strength must be between 0 and 1."""
        # Valid bounds
        assoc = Association(
            id="assoc-1",
            workspace_id="ws-1",
            source_id="mem-1",
            target_id="mem-2",
            relationship="solves",
            strength=0.0,
        )
        assert assoc.strength == 0.0

        assoc = Association(
            id="assoc-1",
            workspace_id="ws-1",
            source_id="mem-1",
            target_id="mem-2",
            relationship="solves",
            strength=1.0,
        )
        assert assoc.strength == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            Association(
                id="assoc-1",
                workspace_id="ws-1",
                source_id="mem-1",
                target_id="mem-2",
                relationship="solves",
                strength=-0.1,
            )

        with pytest.raises(ValidationError):
            Association(
                id="assoc-1",
                workspace_id="ws-1",
                source_id="mem-1",
                target_id="mem-2",
                relationship="solves",
                strength=1.1,
            )

    def test_association_category_property(self):
        """Test Association.category property."""
        assoc = Association(
            id="assoc-1",
            workspace_id="ws-1",
            source_id="mem-1",
            target_id="mem-2",
            relationship="solves",
        )

        assert assoc.category == "solution"


class TestAssociateInput:
    """Test AssociateInput validation."""

    def test_associate_input_required_fields(self):
        """Test required fields for AssociateInput."""
        # Missing source_id
        with pytest.raises(ValidationError):
            AssociateInput(
                target_id="mem-2", relationship="solves"
            )

        # Missing target_id
        with pytest.raises(ValidationError):
            AssociateInput(
                source_id="mem-1", relationship="solves"
            )

        # Missing relationship
        with pytest.raises(ValidationError):
            AssociateInput(source_id="mem-1", target_id="mem-2")

    def test_associate_input_defaults(self):
        """Test AssociateInput default values."""
        input_data = AssociateInput(
            source_id="mem-1",
            target_id="mem-2",
            relationship="solves",
        )

        assert input_data.strength == 0.5
        assert input_data.metadata == {}


class TestGraphQueryInput:
    """Test GraphQueryInput validation."""

    def test_graph_query_max_depth_bounds(self):
        """Test max_depth must be between 1 and 5."""
        # Valid bounds
        input_data = GraphQueryInput(start_memory_id="mem-1", max_depth=1)
        assert input_data.max_depth == 1

        input_data = GraphQueryInput(start_memory_id="mem-1", max_depth=5)
        assert input_data.max_depth == 5

        # Invalid
        with pytest.raises(ValidationError):
            GraphQueryInput(start_memory_id="mem-1", max_depth=0)

        with pytest.raises(ValidationError):
            GraphQueryInput(start_memory_id="mem-1", max_depth=6)

    def test_graph_query_direction_pattern(self):
        """Test direction must match pattern."""
        # Valid directions
        for direction in ["outgoing", "incoming", "both"]:
            input_data = GraphQueryInput(
                start_memory_id="mem-1", direction=direction
            )
            assert input_data.direction == direction

        # Invalid direction
        with pytest.raises(ValidationError):
            GraphQueryInput(start_memory_id="mem-1", direction="invalid")

    def test_graph_query_defaults(self):
        """Test GraphQueryInput default values."""
        input_data = GraphQueryInput(start_memory_id="mem-1")

        assert input_data.max_depth == 3
        assert input_data.direction == "both"
        assert input_data.min_strength == 0.0
        assert input_data.max_paths == 100
        assert input_data.max_nodes == 50


class TestGraphModels:
    """Test GraphPath and GraphQueryResult models."""

    def test_graph_path_model(self):
        """Test GraphPath model creation."""
        assoc = Association(
            id="assoc-1",
            workspace_id="ws-1",
            source_id="mem-1",
            target_id="mem-2",
            relationship="solves",
        )

        path = GraphPath(
            nodes=["mem-1", "mem-2"],
            edges=[assoc],
            total_strength=0.5,
            depth=1,
        )

        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.total_strength == 0.5

    def test_graph_query_result_model(self):
        """Test GraphQueryResult model creation."""
        result = GraphQueryResult(
            paths=[],
            total_paths=0,
            unique_nodes=[],
        )

        assert result.paths == []
        assert result.total_paths == 0
        assert result.query_latency_ms == 0


# ============================================================================
# SESSION MODEL TESTS
# ============================================================================


class TestSessionModel:
    """Test Session model and factory methods."""

    def test_session_create_with_ttl_factory(self):
        """Test Session.create_with_ttl() factory method."""
        now = datetime.now(timezone.utc)
        session = Session.create_with_ttl(
            session_id="sess-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )

        assert session.id == "sess-1"
        assert session.workspace_id == "ws-1"

        # Check expiration is approximately 1 hour from now
        time_diff = (session.expires_at - now).total_seconds()
        assert 3590 <= time_diff <= 3610  # Allow 10 second tolerance

    def test_session_is_expired_property(self):
        """Test Session.is_expired property."""
        # Future expiration - not expired
        session = Session(
            id="sess-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert session.is_expired is False

        # Past expiration - expired
        session = Session(
            id="sess-1",
            workspace_id="ws-1",
            tenant_id="default_tenant",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert session.is_expired is True


class TestWorkingMemory:
    """Test WorkingMemory model validation."""

    def test_working_memory_key_not_empty(self):
        """Test that working memory key cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            WorkingMemory(
                session_id="sess-1",
                key="",
                value="test",
            )

        errors = exc_info.value.errors()
        assert any("key" in str(e.get("loc")) for e in errors)

    def test_working_memory_key_whitespace_stripped(self):
        """Test that key whitespace is stripped."""
        entry = WorkingMemory(
            session_id="sess-1",
            key="  valid_key  ",
            value="test",
        )
        assert entry.key == "valid_key"


class TestSessionBriefingModels:
    """Test session briefing related models."""

    def test_session_briefing_model(self):
        """Test SessionBriefing model."""
        briefing = SessionBriefing(
            workspace_summary={"total_memories": 100},
            recent_activity=[],
            open_threads=[],
            contradictions_detected=[],
        )

        assert briefing.workspace_summary["total_memories"] == 100
        assert briefing.recent_activity == []

    def test_workspace_summary_model(self):
        """Test WorkspaceSummary model."""
        summary = WorkspaceSummary(
            total_memories=150,
            recent_memories=10,
            active_topics=["python", "testing"],
            total_categories=5,
            total_associations=200,
        )

        assert summary.total_memories == 150
        assert len(summary.active_topics) == 2

    def test_activity_summary_model(self):
        """Test ActivitySummary model."""
        summary = ActivitySummary(
            timestamp=datetime.now(timezone.utc),
            summary="Implemented new feature",
            memories_created=5,
            key_decisions=["Use SQLite", "Add caching"],
        )

        assert summary.memories_created == 5
        assert len(summary.key_decisions) == 2

    def test_open_thread_model(self):
        """Test OpenThread model."""
        thread = OpenThread(
            topic="Authentication refactor",
            status="in_progress",
            last_activity=datetime.now(timezone.utc),
            key_memories=["mem-1", "mem-2"],
        )

        assert thread.status == "in_progress"
        assert len(thread.key_memories) == 2

    def test_contradiction_model(self):
        """Test Contradiction model."""
        contradiction = Contradiction(
            memory_a="mem-1",
            memory_b="mem-2",
            relationship="contradicts",
        )

        assert contradiction.needs_resolution is True
        assert contradiction.relationship == "contradicts"


# ============================================================================
# WORKSPACE MODEL TESTS
# ============================================================================


class TestWorkspaceModel:
    """Test Workspace model validation."""

    def test_workspace_name_not_empty(self):
        """Test that workspace name cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            Workspace(
                id="ws-1",
                tenant_id="tenant-1",
                name="",
            )

        errors = exc_info.value.errors()
        assert any("name" in str(e.get("loc")) for e in errors)

    def test_workspace_name_whitespace_stripped(self):
        """Test that workspace name whitespace is stripped."""
        workspace = Workspace(
            id="ws-1",
            tenant_id="tenant-1",
            name="  My Workspace  ",
        )
        assert workspace.name == "My Workspace"


class TestContextModel:
    """Test Context model validation."""

    def test_context_name_not_empty(self):
        """Test that context name cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            Context(
                id="ctx-1",
                workspace_id="ws-1",
                name="",
            )

        errors = exc_info.value.errors()
        assert any("name" in str(e.get("loc")) for e in errors)

    def test_context_name_whitespace_stripped(self):
        """Test that context name whitespace is stripped."""
        ctx = Context(
            id="ctx-1",
            workspace_id="ws-1",
            name="  My Context  ",
        )
        assert ctx.name == "My Context"


class TestWorkspaceSettings:
    """Test WorkspaceSettings model."""

    def test_workspace_settings_defaults(self):
        """Test WorkspaceSettings default values."""
        settings = WorkspaceSettings()

        # Retention
        assert settings.default_importance == 0.5
        assert settings.decay_enabled is True
        assert settings.decay_rate == 0.01

        # Auto-remember
        assert settings.auto_remember_enabled is False
        assert settings.auto_remember_min_importance == 0.6
        assert settings.auto_remember_exclude_patterns == []

        # Embeddings
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536

        # Storage tiers
        assert settings.hot_tier_days == 7
        assert settings.warm_tier_days == 90
        assert settings.enable_cold_tier is False


class TestContextSettings:
    """Test ContextSettings model."""

    def test_context_settings_inheritance_default(self):
        """Test ContextSettings inheritance default."""
        settings = ContextSettings()

        assert settings.inherit_workspace_settings is True
        assert settings.auto_remember_enabled is None
        assert settings.decay_enabled is None


