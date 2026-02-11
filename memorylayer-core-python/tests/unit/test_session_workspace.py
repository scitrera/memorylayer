"""
Unit tests for Session and Workspace models.

Tests:
- Session TTL and expiration
- WorkingMemory key-value storage
- Session briefing models
- Workspace creation and validation
- Context creation and validation
- WorkspaceSettings default values
- ContextSettings inheritance
- Memory modes (explicit vs auto_remember)
"""
import pytest
from datetime import datetime, timedelta, timezone
from memorylayer_server.models.session import (
    Session,
    WorkingMemory,
    SessionBriefing,
    WorkspaceSummary,
    ActivitySummary,
    OpenThread,
    Contradiction,
)
from memorylayer_server.models.workspace import (
    Workspace,
    Context,
    WorkspaceSettings,
    ContextSettings,
)



class TestSession:
    """Tests for Session model (Section 3.10, 5.5)."""

    def test_create_with_ttl_creates_correct_expiration(self):
        """Test Session.create_with_ttl() creates session with correct expiration."""
        ttl_seconds = 7200  # 2 hours
        session = Session.create_with_ttl(
            session_id="sess_test_123",
            workspace_id="ws_test",
            ttl_seconds=ttl_seconds,
            tenant_id="default_tenant",
        )

        assert session.id == "sess_test_123"
        assert session.workspace_id == "ws_test"

        # Check expiration is approximately ttl_seconds from now
        expected_expiration = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        # Allow 5 second tolerance for test execution time
        assert abs((session.expires_at - expected_expiration).total_seconds()) < 5

    def test_is_expired_returns_true_after_ttl(self):
        """Test Session.is_expired returns True after TTL."""
        # Create session that expired 1 hour ago
        session = Session.create_with_ttl(
            session_id="sess_expired",
            workspace_id="ws_test",
            ttl_seconds=-3600,  # Negative TTL = already expired
            tenant_id="default_tenant",
        )

        assert session.is_expired is True

    def test_is_expired_returns_false_before_ttl(self):
        """Test Session.is_expired returns False before TTL."""
        # Create session that expires in 1 hour
        session = Session.create_with_ttl(
            session_id="sess_active",
            workspace_id="ws_test",
            ttl_seconds=3600,
            tenant_id="default_tenant",
        )

        assert session.is_expired is False

    def test_session_with_metadata(self):
        """Test Session can store metadata."""
        metadata = {
            "client": "python-sdk",
            "version": "1.0.0",
            "user_agent": "test-agent",
        }
        session = Session.create_with_ttl(
            session_id="sess_meta",
            workspace_id="ws_test",
            ttl_seconds=3600,
            tenant_id="default_tenant",
            metadata=metadata,
        )

        assert session.metadata == metadata
        assert session.metadata["client"] == "python-sdk"

    def test_session_with_user_id(self):
        """Test Session can be scoped to a user."""
        session = Session.create_with_ttl(
            session_id="sess_user",
            workspace_id="ws_test",
            ttl_seconds=3600,
            tenant_id="default_tenant",
            user_id="user_123",
        )

        assert session.user_id == "user_123"


class TestWorkingMemoryModel:
    """Tests for WorkingMemory model (Section 3.10)."""

    def test_working_memory_key_validation_empty_raises(self):
        """Test WorkingMemory raises error for empty key."""
        with pytest.raises(ValueError, match="Context key cannot be empty"):
            WorkingMemory(
                session_id="sess_test",
                key="",
                value="some value",
            )

    def test_working_memory_key_validation_whitespace_raises(self):
        """Test WorkingMemory raises error for whitespace-only key."""
        with pytest.raises(ValueError, match="Context key cannot be empty"):
            WorkingMemory(
                session_id="sess_test",
                key="   ",
                value="some value",
            )

    def test_working_memory_value_storage_json_serializable(self):
        """Test WorkingMemory stores JSON-serializable values."""
        # String value
        wm1 = WorkingMemory(
            session_id="sess_test",
            key="string_key",
            value="string_value",
        )
        assert wm1.value == "string_value"

        # Dict value
        wm2 = WorkingMemory(
            session_id="sess_test",
            key="dict_key",
            value={"nested": "data", "count": 42},
        )
        assert wm2.value == {"nested": "data", "count": 42}

        # List value
        wm3 = WorkingMemory(
            session_id="sess_test",
            key="list_key",
            value=[1, 2, 3, "four"],
        )
        assert wm3.value == [1, 2, 3, "four"]

        # Number value
        wm4 = WorkingMemory(
            session_id="sess_test",
            key="number_key",
            value=123.45,
        )
        assert wm4.value == 123.45

        # Boolean value
        wm5 = WorkingMemory(
            session_id="sess_test",
            key="bool_key",
            value=True,
        )
        assert wm5.value is True

    def test_working_memory_with_ttl_override(self):
        """Test WorkingMemory can override session TTL."""
        wm = WorkingMemory(
            session_id="sess_test",
            key="short_lived",
            value="expires soon",
            ttl_seconds=300,  # 5 minutes
        )
        assert wm.ttl_seconds == 300

    def test_working_memory_key_trimmed(self):
        """Test WorkingMemory trims whitespace from keys."""
        wm = WorkingMemory(
            session_id="sess_test",
            key="  trimmed_key  ",
            value="value",
        )
        assert wm.key == "trimmed_key"


class TestSessionBriefing:
    """Tests for SessionBriefing model (Section 5.5)."""

    def test_session_briefing_creation(self):
        """Test SessionBriefing model creation."""
        briefing = SessionBriefing(
            workspace_summary={
                "total_memories": 150,
                "recent_memories": 12,
            },
            recent_activity=[
                {"summary": "Added 5 new memories", "timestamp": datetime.now(timezone.utc).isoformat()}
            ],
            open_threads=[
                {"topic": "API refactoring", "status": "in_progress"}
            ],
            contradictions_detected=[
                {"memory_a": "mem_1", "memory_b": "mem_2"}
            ],
        )

        assert briefing.workspace_summary["total_memories"] == 150
        assert len(briefing.recent_activity) == 1
        assert len(briefing.open_threads) == 1
        assert len(briefing.contradictions_detected) == 1


class TestWorkspaceSummary:
    """Tests for WorkspaceSummary model (Section 5.5)."""

    def test_workspace_summary_with_all_fields(self):
        """Test WorkspaceSummary model with all fields."""
        summary = WorkspaceSummary(
            total_memories=250,
            recent_memories=15,
            active_topics=["python", "testing", "api"],
            total_categories=8,
            total_associations=42,
        )

        assert summary.total_memories == 250
        assert summary.recent_memories == 15
        assert "python" in summary.active_topics
        assert summary.total_categories == 8
        assert summary.total_associations == 42

    def test_workspace_summary_defaults(self):
        """Test WorkspaceSummary default values."""
        summary = WorkspaceSummary()

        assert summary.total_memories == 0
        assert summary.recent_memories == 0
        assert summary.active_topics == []
        assert summary.total_categories == 0
        assert summary.total_associations == 0


class TestActivitySummary:
    """Tests for ActivitySummary model (Section 5.5)."""

    def test_activity_summary_creation(self):
        """Test ActivitySummary model creation."""
        timestamp = datetime.now(timezone.utc)
        activity = ActivitySummary(
            timestamp=timestamp,
            summary="User added authentication memories",
            memories_created=3,
            key_decisions=["Use JWT tokens", "Add rate limiting"],
        )

        assert activity.timestamp == timestamp
        assert activity.summary == "User added authentication memories"
        assert activity.memories_created == 3
        assert len(activity.key_decisions) == 2


class TestOpenThread:
    """Tests for OpenThread model (Section 5.5)."""

    def test_open_thread_with_status_values(self):
        """Test OpenThread model with status values."""
        # Test all valid status values
        for status in ["in_progress", "blocked", "waiting"]:
            thread = OpenThread(
                topic=f"Test thread {status}",
                status=status,
                last_activity=datetime.now(timezone.utc),
                key_memories=["mem_1", "mem_2"],
            )
            assert thread.status == status

    def test_open_thread_with_key_memories(self):
        """Test OpenThread tracks key memory IDs."""
        thread = OpenThread(
            topic="Feature implementation",
            status="in_progress",
            last_activity=datetime.now(timezone.utc),
            key_memories=["mem_123", "mem_456", "mem_789"],
        )

        assert len(thread.key_memories) == 3
        assert "mem_123" in thread.key_memories


class TestContradiction:
    """Tests for Contradiction model (Section 5.5)."""

    def test_contradiction_creation(self):
        """Test Contradiction model creation."""
        contradiction = Contradiction(
            memory_a="mem_old",
            memory_b="mem_new",
            relationship="contradicts",
        )

        assert contradiction.memory_a == "mem_old"
        assert contradiction.memory_b == "mem_new"
        assert contradiction.relationship == "contradicts"
        assert contradiction.needs_resolution is True
        assert isinstance(contradiction.detected_at, datetime)

    def test_contradiction_needs_resolution_flag(self):
        """Test Contradiction needs_resolution flag."""
        # Default should be True
        c1 = Contradiction(
            memory_a="mem_1",
            memory_b="mem_2",
            relationship="contradicts",
        )
        assert c1.needs_resolution is True

        # Can be explicitly set to False
        c2 = Contradiction(
            memory_a="mem_3",
            memory_b="mem_4",
            relationship="contradicts",
            needs_resolution=False,
        )
        assert c2.needs_resolution is False


class TestWorkspace:
    """Tests for Workspace model (Section 3.5, 3.11)."""

    def test_workspace_creation(self):
        """Test Workspace creation with required fields."""
        workspace = Workspace(
            id="ws_test_123",
            tenant_id="tenant_abc",
            name="Test Workspace",
        )

        assert workspace.id == "ws_test_123"
        assert workspace.tenant_id == "tenant_abc"
        assert workspace.name == "Test Workspace"
        assert isinstance(workspace.created_at, datetime)
        assert isinstance(workspace.updated_at, datetime)

    def test_workspace_name_validation_empty_raises(self):
        """Test Workspace raises error for empty name."""
        with pytest.raises(ValueError, match="Workspace name cannot be empty"):
            Workspace(
                id="ws_test",
                tenant_id="tenant_test",
                name="",
            )

    def test_workspace_name_validation_whitespace_raises(self):
        """Test Workspace raises error for whitespace-only name."""
        with pytest.raises(ValueError, match="Workspace name cannot be empty"):
            Workspace(
                id="ws_test",
                tenant_id="tenant_test",
                name="   ",
            )

    def test_workspace_name_trimmed(self):
        """Test Workspace trims whitespace from name."""
        workspace = Workspace(
            id="ws_test",
            tenant_id="tenant_test",
            name="  Trimmed Name  ",
        )
        assert workspace.name == "Trimmed Name"

    def test_workspace_settings_dictionary(self):
        """Test Workspace stores settings as dictionary."""
        settings = {
            "default_importance": 0.7,
            "decay_enabled": True,
            "auto_remember_enabled": False,
        }
        workspace = Workspace(
            id="ws_test",
            tenant_id="tenant_test",
            name="Settings Test",
            settings=settings,
        )

        assert workspace.settings == settings
        assert workspace.settings["default_importance"] == 0.7
        assert workspace.settings["decay_enabled"] is True


class TestContextModel:
    """Tests for Context model (Section 3.5)."""

    def test_context_creation_within_workspace(self):
        """Test Context creation within workspace."""
        ctx = Context(
            id="ctx_test_123",
            workspace_id="ws_parent",
            name="Test Context",
            description="A test context",
        )

        assert ctx.id == "ctx_test_123"
        assert ctx.workspace_id == "ws_parent"
        assert ctx.name == "Test Context"
        assert ctx.description == "A test context"
        assert isinstance(ctx.created_at, datetime)

    def test_context_name_validation_empty_raises(self):
        """Test Context raises error for empty name."""
        with pytest.raises(ValueError, match="Context name cannot be empty"):
            Context(
                id="ctx_test",
                workspace_id="ws_test",
                name="",
            )

    def test_context_name_validation_whitespace_raises(self):
        """Test Context raises error for whitespace-only name."""
        with pytest.raises(ValueError, match="Context name cannot be empty"):
            Context(
                id="ctx_test",
                workspace_id="ws_test",
                name="   ",
            )

    def test_context_name_trimmed(self):
        """Test Context trims whitespace from name."""
        ctx = Context(
            id="ctx_test",
            workspace_id="ws_test",
            name="  Trimmed Context  ",
        )
        assert ctx.name == "Trimmed Context"

    def test_context_settings_dictionary(self):
        """Test Context stores settings dictionary."""
        settings = {
            "inherit_workspace_settings": False,
            "auto_remember_enabled": True,
        }
        ctx = Context(
            id="ctx_test",
            workspace_id="ws_test",
            name="Settings Context",
            settings=settings,
        )

        assert ctx.settings == settings
        assert ctx.settings["inherit_workspace_settings"] is False


class TestWorkspaceSettings:
    """Tests for WorkspaceSettings model (Section 3.11)."""

    def test_workspace_settings_default_values(self):
        """Test WorkspaceSettings default values."""
        settings = WorkspaceSettings()

        # Retention defaults
        assert settings.default_importance == 0.5
        assert settings.decay_enabled is True
        assert settings.decay_rate == 0.01

        # Auto-remember defaults
        assert settings.auto_remember_enabled is False
        assert settings.auto_remember_min_importance == 0.6
        assert settings.auto_remember_exclude_patterns == []

        # Embedding defaults
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536

        # Storage tier defaults
        assert settings.hot_tier_days == 7
        assert settings.warm_tier_days == 90
        assert settings.enable_cold_tier is False

    def test_workspace_settings_custom_values(self):
        """Test WorkspaceSettings with custom values."""
        settings = WorkspaceSettings(
            default_importance=0.7,
            decay_enabled=False,
            decay_rate=0.05,
            auto_remember_enabled=True,
            auto_remember_min_importance=0.8,
            auto_remember_exclude_patterns=["debug", "test"],
            embedding_model="custom-model",
            embedding_dimensions=768,
            hot_tier_days=14,
            warm_tier_days=180,
            enable_cold_tier=True,
        )

        assert settings.default_importance == 0.7
        assert settings.decay_enabled is False
        assert settings.decay_rate == 0.05
        assert settings.auto_remember_enabled is True
        assert settings.auto_remember_min_importance == 0.8
        assert "debug" in settings.auto_remember_exclude_patterns
        assert settings.embedding_model == "custom-model"
        assert settings.embedding_dimensions == 768
        assert settings.hot_tier_days == 14
        assert settings.warm_tier_days == 180
        assert settings.enable_cold_tier is True

    def test_workspace_settings_validation_ranges(self):
        """Test WorkspaceSettings field validations."""
        # Valid at boundaries
        settings = WorkspaceSettings(
            default_importance=0.0,  # Min
            decay_rate=1.0,  # Max
            auto_remember_min_importance=1.0,  # Max
        )
        assert settings.default_importance == 0.0
        assert settings.decay_rate == 1.0

        # Invalid: importance out of range
        with pytest.raises(ValueError):
            WorkspaceSettings(default_importance=1.5)

        with pytest.raises(ValueError):
            WorkspaceSettings(default_importance=-0.1)


class TestContextSettings:
    """Tests for ContextSettings model (Section 3.11)."""

    def test_context_settings_inheritance_flag(self):
        """Test ContextSettings inheritance flag."""
        # Default should inherit
        settings1 = ContextSettings()
        assert settings1.inherit_workspace_settings is True

        # Can be explicitly disabled
        settings2 = ContextSettings(inherit_workspace_settings=False)
        assert settings2.inherit_workspace_settings is False

    def test_context_settings_overrides(self):
        """Test ContextSettings can override workspace settings."""
        settings = ContextSettings(
            inherit_workspace_settings=False,
            auto_remember_enabled=True,
            decay_enabled=False,
        )

        assert settings.inherit_workspace_settings is False
        assert settings.auto_remember_enabled is True
        assert settings.decay_enabled is False

    def test_context_settings_none_values_when_inheriting(self):
        """Test ContextSettings overrides are None when inheriting."""
        settings = ContextSettings(inherit_workspace_settings=True)

        # Overrides should be None when inheriting
        assert settings.auto_remember_enabled is None
        assert settings.decay_enabled is None


class TestMemoryModes:
    """Tests for memory modes (Section 3.11)."""

    def test_explicit_mode_default(self):
        """Test explicit mode (memories only stored on explicit call)."""
        # Explicit mode is when auto_remember_enabled=False (default)
        settings = WorkspaceSettings()

        assert settings.auto_remember_enabled is False
        # In explicit mode, memories are only created via explicit remember() calls

    def test_auto_remember_settings(self):
        """Test auto_remember settings in WorkspaceSettings."""
        settings = WorkspaceSettings(
            auto_remember_enabled=True,
            auto_remember_min_importance=0.75,
            auto_remember_exclude_patterns=["temp_*", "debug_*", "test_*"],
        )

        assert settings.auto_remember_enabled is True
        assert settings.auto_remember_min_importance == 0.75
        assert len(settings.auto_remember_exclude_patterns) == 3
        assert "temp_*" in settings.auto_remember_exclude_patterns
        assert "debug_*" in settings.auto_remember_exclude_patterns
        assert "test_*" in settings.auto_remember_exclude_patterns

    def test_auto_remember_min_importance_validation(self):
        """Test auto_remember_min_importance must be between 0 and 1."""
        # Valid values
        settings1 = WorkspaceSettings(auto_remember_min_importance=0.0)
        assert settings1.auto_remember_min_importance == 0.0

        settings2 = WorkspaceSettings(auto_remember_min_importance=1.0)
        assert settings2.auto_remember_min_importance == 1.0

        # Invalid values
        with pytest.raises(ValueError):
            WorkspaceSettings(auto_remember_min_importance=-0.1)

        with pytest.raises(ValueError):
            WorkspaceSettings(auto_remember_min_importance=1.5)

    def test_context_can_override_auto_remember(self):
        """Test ContextSettings can override workspace auto_remember settings."""
        # Workspace has auto_remember disabled
        workspace_settings = WorkspaceSettings(auto_remember_enabled=False)

        # Context can enable it independently
        context_settings = ContextSettings(
            inherit_workspace_settings=False,
            auto_remember_enabled=True,
        )

        assert workspace_settings.auto_remember_enabled is False
        assert context_settings.auto_remember_enabled is True
