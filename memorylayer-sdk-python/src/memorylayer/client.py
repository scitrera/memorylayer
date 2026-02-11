"""Main client for MemoryLayer.ai SDK."""

import logging
from typing import Any, Optional, Union

import httpx
from pydantic import TypeAdapter

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    MemoryLayerError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    Association,
    Memory,
    RecallResult,
    ReflectResult,
    Session,
    SessionBriefing,
    Workspace,
)
from .types import (
    MemorySubtype,
    MemoryType,
    RecallMode,
    RelationshipType,
    SearchTolerance,
)

logger = logging.getLogger(__name__)


def _to_value(v: Any) -> Any:
    """Extract string value from an enum member, or return string as-is."""
    return v.value if hasattr(v, "value") else v


class MemoryLayerClient:
    """
    Python client for MemoryLayer.ai API.

    Usage:
        async with MemoryLayerClient(
            base_url="https://api.memorylayer.ai",
            api_key="your-api-key",
            workspace_id="ws_123"
        ) as client:
            # Store a memory
            memory = await client.remember(
                content="User prefers Python",
                type=MemoryType.SEMANTIC,
                importance=0.8
            )

            # Search memories
            results = await client.recall("coding preferences")

            # Reflect on memories
            reflection = await client.reflect("summarize user preferences")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:61001",
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize MemoryLayer client.

        Args:
            base_url: API base URL (default: http://localhost:61001)
            api_key: API key for authentication
            workspace_id: Default workspace ID for operations
            session_id: Session ID for session-based workspace resolution
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "MemoryLayerClient":
        """Async context manager entry."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.session_id:
            headers["X-Session-ID"] = self.session_id

        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/v1",
            headers=headers,
            timeout=self.timeout,
        )
        return self

    def set_session(self, session_id: str) -> None:
        """
        Set the active session ID.

        All subsequent requests will include this session ID in the
        X-Session-ID header, enabling session-based workspace resolution.

        Note: If called after entering the context manager, updates
        the underlying client headers.

        Args:
            session_id: Session ID to use
        """
        self.session_id = session_id
        if self._client:
            self._client.headers["X-Session-ID"] = session_id

    def clear_session(self) -> None:
        """
        Clear the active session ID.
        """
        self.session_id = None
        if self._client and "X-Session-ID" in self._client.headers:
            del self._client.headers["X-Session-ID"]

    def get_session_id(self) -> Optional[str]:
        """
        Get the current session ID, if any.

        Returns:
            Current session ID or None
        """
        return self.session_id

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async with context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make HTTP request with error handling.

        Args:
            method: HTTP method
            path: API path
            json: JSON body
            params: Query parameters

        Returns:
            Response JSON

        Raises:
            AuthenticationError: Authentication failed (401)
            NotFoundError: Resource not found (404)
            ValidationError: Validation failed (422)
            RateLimitError: Rate limit exceeded (429)
            ServerError: Server error (5xx)
            MemoryLayerError: Other errors
        """
        client = self._ensure_client()

        try:
            response = await client.request(method, path, json=json, params=params)

            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError(response.json().get("detail", "Authentication failed"))
            elif response.status_code == 403:
                raise AuthorizationError(response.json().get("detail", "Authorization denied"))
            elif response.status_code == 404:
                raise NotFoundError(response.json().get("detail", "Resource not found"))
            elif response.status_code == 422:
                raise ValidationError(response.json().get("detail", "Validation error"))
            elif response.status_code == 429:
                raise RateLimitError(response.json().get("detail", "Rate limit exceeded"))
            elif response.status_code >= 500:
                raise ServerError(
                    response.json().get("detail", "Server error"),
                    status_code=response.status_code,
                )
            elif response.status_code >= 400:
                raise MemoryLayerError(
                    response.json().get("detail", "Request failed"),
                    status_code=response.status_code,
                )

            response.raise_for_status()

            # Handle No Content responses
            if response.status_code == 204:
                return {}

            return response.json()

        except httpx.TimeoutException as e:
            raise MemoryLayerError(f"Request timeout: {e}") from e
        except httpx.HTTPError as e:
            raise MemoryLayerError(f"HTTP error: {e}") from e

    # Core memory operations

    async def remember(
        self,
        content: str,
        type: Optional[Union[str, MemoryType]] = None,
        subtype: Optional[Union[str, MemorySubtype]] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        space_id: Optional[str] = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: Memory content to store
            type: Cognitive memory type (episodic, semantic, procedural, working)
            subtype: Domain subtype (Solution, Problem, etc.)
            importance: Importance score 0.0-1.0 (default: 0.5)
            tags: Tags for categorization
            metadata: Additional metadata
            space_id: Optional memory space ID

        Returns:
            Created memory

        Example:
            memory = await client.remember(
                content="User prefers concise code comments",
                type=MemoryType.SEMANTIC,
                subtype=MemorySubtype.PREFERENCE,
                importance=0.8,
                tags=["preferences", "coding-style"]
            )
        """
        payload = {
            "content": content,
            "importance": importance,
        }
        if type:
            payload["type"] = _to_value(type)
        if subtype:
            payload["subtype"] = _to_value(subtype)
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata
        if space_id:
            payload["space_id"] = space_id

        data = await self._request("POST", "/memories", json=payload)
        return Memory(**data)

    async def recall(
        self,
        query: str,
        types: Optional[list[Union[str, MemoryType]]] = None,
        subtypes: Optional[list[Union[str, MemorySubtype]]] = None,
        tags: Optional[list[str]] = None,
        mode: Optional[Union[str, RecallMode]] = None,
        limit: int = 10,
        min_relevance: Optional[float] = None,
        recency_weight: Optional[float] = None,
        tolerance: Optional[Union[str, SearchTolerance]] = None,
        include_associations: Optional[bool] = None,
        traverse_depth: Optional[int] = None,
        max_expansion: Optional[int] = None,
    ) -> RecallResult:
        """
        Search memories by semantic query.

        Args:
            query: Natural language search query
            types: Filter by cognitive memory types
            subtypes: Filter by domain subtypes
            tags: Filter by tags
            mode: Retrieval mode (rag, llm, hybrid). None = server default.
            limit: Maximum memories to return (default: 10)
            min_relevance: Minimum relevance score 0.0-1.0 (None = server default)
            recency_weight: Weight for recency in ranking 0.0-1.0 (None = server default)
            tolerance: Search tolerance (loose, moderate, strict). None = server default.
            include_associations: Include linked memories (None = server default)
            traverse_depth: Multi-hop graph traversal depth (None = server default)
            max_expansion: Max memories discovered via graph expansion (None = server default)

        Returns:
            Recall results with memories

        Example:
            results = await client.recall(
                query="what are the user's coding preferences?",
                types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                limit=5,
                min_relevance=0.7
            )
        """
        payload: dict = {
            "query": query,
            "limit": limit,
        }
        if mode is not None:
            payload["mode"] = _to_value(mode)
        if tolerance is not None:
            payload["tolerance"] = _to_value(tolerance)
        if include_associations is not None:
            payload["include_associations"] = include_associations
        if traverse_depth is not None:
            payload["traverse_depth"] = traverse_depth
        if min_relevance is not None:
            payload["min_relevance"] = min_relevance
        if recency_weight is not None:
            payload["recency_weight"] = recency_weight
        if max_expansion is not None:
            payload["max_expansion"] = max_expansion
        if types:
            payload["types"] = [_to_value(t) for t in types]
        if subtypes:
            payload["subtypes"] = [_to_value(s) for s in subtypes]
        if tags:
            payload["tags"] = tags

        data = await self._request("POST", "/memories/recall", json=payload)

        # Parse memories
        memories_adapter = TypeAdapter(list[Memory])
        memories = memories_adapter.validate_python(data.get("memories", []))

        return RecallResult(
            memories=memories,
            total_count=data.get("total_count", len(memories)),
            query_tokens=data.get("query_tokens"),
            search_latency_ms=data.get("search_latency_ms"),
        )

    async def reflect(
        self,
        query: str,
        max_tokens: int = 500,
        include_sources: bool = True,
    ) -> ReflectResult:
        """
        Synthesize and summarize memories.

        Args:
            query: What to reflect on
            max_tokens: Maximum tokens in reflection (default: 500)
            include_sources: Include source memory IDs (default: True)

        Returns:
            Reflection result with synthesis

        Example:
            reflection = await client.reflect(
                query="summarize everything about user's development workflow",
                max_tokens=300
            )
        """
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "include_sources": include_sources,
        }

        data = await self._request("POST", "/memories/reflect", json=payload)
        return ReflectResult(**data)

    async def forget(
        self,
        memory_id: str,
        hard: bool = False,
    ) -> bool:
        """
        Delete or soft-delete a memory.

        Args:
            memory_id: ID of memory to forget
            hard: Permanently delete (default: False for soft delete)

        Returns:
            True if successful

        Example:
            await client.forget("mem_123", hard=False)
        """
        params = {"hard": "true" if hard else "false"}
        await self._request("DELETE", f"/memories/{memory_id}", params=params)
        return True

    async def get_memory(self, memory_id: str) -> Memory:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object

        Example:
            memory = await client.get_memory("mem_123")
        """
        data = await self._request("GET", f"/memories/{memory_id}")
        return Memory(**data)

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Memory:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID
            content: New content (optional)
            importance: New importance score (optional)
            tags: New tags (optional)
            metadata: New metadata (optional)

        Returns:
            Updated memory

        Example:
            memory = await client.update_memory(
                "mem_123",
                importance=0.9,
                tags=["preferences", "high-priority"]
            )
        """
        payload = {}
        if content is not None:
            payload["content"] = content
        if importance is not None:
            payload["importance"] = importance
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        data = await self._request("PUT", f"/memories/{memory_id}", json=payload)
        return Memory(**data)

    # Association methods

    async def associate(
        self,
        source_id: str,
        target_id: str,
        relationship: Union[str, RelationshipType],
        strength: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Association:
        """
        Link two memories with a relationship.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            relationship: Relationship type (string or RelationshipType enum)
            strength: Relationship strength 0.0-1.0 (default: 0.5)
            metadata: Additional metadata

        Returns:
            Created association

        Example:
            assoc = await client.associate(
                source_id="mem_123",
                target_id="mem_456",
                relationship=RelationshipType.SOLVES,
                strength=0.9
            )
        """
        rel_value = _to_value(relationship)
        payload = {
            "target_id": target_id,
            "relationship": rel_value,
            "strength": strength,
        }
        if metadata:
            payload["metadata"] = metadata

        data = await self._request("POST", f"/memories/{source_id}/associate", json=payload)
        return Association(**data)

    async def get_associations(
        self,
        memory_id: str,
        direction: str = "both",
    ) -> list[Association]:
        """
        Get associations for a memory.

        Args:
            memory_id: Memory ID
            direction: "outgoing", "incoming", or "both" (default: "both")

        Returns:
            List of associations

        Example:
            associations = await client.get_associations("mem_123")
        """
        params = {"direction": direction}
        data = await self._request("GET", f"/memories/{memory_id}/associations", params=params)

        associations_adapter = TypeAdapter(list[Association])
        return associations_adapter.validate_python(data.get("associations", []))

    # Session methods

    async def create_session(
        self,
        ttl_seconds: int = 3600,
        workspace_id: Optional[str] = None,
        context_id: Optional[str] = None,
        auto_set_session: bool = True,
    ) -> Session:
        """
        Create a new working memory session.

        Workspaces and contexts are auto-created if they don't exist,
        enabling a "just works" experience for new projects.

        Args:
            ttl_seconds: Time to live in seconds (default: 3600 = 1 hour)
            workspace_id: Workspace ID (auto-created if doesn't exist)
            context_id: Context ID (defaults to _default, auto-created if needed)
            auto_set_session: If True (default), automatically set this session
                              as the active session for subsequent requests

        Returns:
            Created session

        Example:
            session = await client.create_session(ttl_seconds=7200)
            # Or with custom workspace (auto-created from git repo name):
            session = await client.create_session(workspace_id="my-project")
        """
        payload: dict[str, Any] = {"ttl_seconds": ttl_seconds}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if context_id:
            payload["context_id"] = context_id
        data = await self._request("POST", "/sessions", json=payload)
        # Handle response format: {session: {...}} or direct session object
        if "session" in data:
            session = Session(**data["session"])
        else:
            session = Session(**data)

        if auto_set_session:
            self.set_session(session.id)

        return session

    async def get_session(self, session_id: str) -> Session:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object
        """
        data = await self._request("GET", f"/sessions/{session_id}")
        return Session(**data)

    async def set_context(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Set a context value in a session.

        Args:
            session_id: Session ID
            key: Context key
            value: Context value (any JSON-serializable object)
            ttl_seconds: Optional TTL for this key

        Example:
            await client.set_context(
                "sess_123",
                "current_file",
                {"path": "auth.py", "line": 42}
            )
        """
        payload = {
            "key": key,
            "value": value,
        }
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds

        await self._request("POST", f"/sessions/{session_id}/memory", json=payload)

    async def get_context(
        self,
        session_id: str,
        keys: list[str],
    ) -> dict[str, Any]:
        """
        Get context values from a session.

        Args:
            session_id: Session ID
            keys: List of keys to retrieve

        Returns:
            Dictionary of key-value pairs

        Example:
            context = await client.get_context(
                "sess_123",
                ["current_file", "user_intent"]
            )
        """
        params = {"key": ",".join(keys)}
        data = await self._request("GET", f"/sessions/{session_id}/memory", params=params)
        return data

    async def get_briefing(self, lookback_hours: int = 24) -> SessionBriefing:
        """
        Get a session briefing with recent activity and context.

        Args:
            lookback_hours: How far back to look for activity (default: 24)

        Returns:
            Session briefing

        Example:
            briefing = await client.get_briefing(lookback_hours=48)
        """
        params = {"lookback_hours": lookback_hours}
        data = await self._request("GET", "/sessions/briefing", params=params)
        return SessionBriefing(**data)

    # Workspace methods

    async def create_workspace(self, name: str) -> Workspace:
        """
        Create a new workspace.

        Args:
            name: Workspace name

        Returns:
            Created workspace

        Example:
            workspace = await client.create_workspace("my-project")
        """
        payload = {"name": name}
        data = await self._request("POST", "/workspaces", json=payload)
        return Workspace(**data)

    async def get_workspace(self, workspace_id: Optional[str] = None) -> Workspace:
        """
        Get workspace details.

        Args:
            workspace_id: Workspace ID (uses default if not provided)

        Returns:
            Workspace object

        Example:
            workspace = await client.get_workspace()
        """
        ws_id = workspace_id or self.workspace_id
        if not ws_id:
            raise ValueError("workspace_id must be provided or set on client")

        data = await self._request("GET", f"/workspaces/{ws_id}")
        return Workspace(**data)

    async def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> Workspace:
        """
        Update an existing workspace.

        Args:
            workspace_id: Workspace ID
            name: New workspace name (optional)
            settings: New workspace settings (optional)

        Returns:
            Updated workspace

        Example:
            workspace = await client.update_workspace(
                "ws_123",
                name="New Name",
                settings={"key": "value"}
            )
        """
        payload = {}
        if name is not None:
            payload["name"] = name
        if settings is not None:
            payload["settings"] = settings

        data = await self._request("PUT", f"/workspaces/{workspace_id}", json=payload)
        return Workspace(**data.get("workspace", data))

    async def create_context(
        self,
        workspace_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Create a context within a workspace.

        Contexts provide logical grouping of memories (e.g., by project, topic).

        Args:
            workspace_id: Parent workspace ID
            name: Context name
            description: Context description (optional)
            settings: Context settings (optional)

        Returns:
            Created context

        Example:
            context = await client.create_context(
                "ws_123",
                name="project-alpha",
                description="Memories for Project Alpha"
            )
        """
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if settings is not None:
            payload["settings"] = settings

        data = await self._request("POST", f"/workspaces/{workspace_id}/contexts", json=payload)
        return data.get("context", data)

    async def list_contexts(self, workspace_id: str) -> list[dict[str, Any]]:
        """
        List all contexts in a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            List of contexts

        Example:
            contexts = await client.list_contexts("ws_123")
        """
        data = await self._request("GET", f"/workspaces/{workspace_id}/contexts")
        return data.get("contexts", [])

    async def get_workspace_schema(self, workspace_id: str) -> dict[str, Any]:
        """
        Get workspace schema including relationship types and memory subtypes.

        Args:
            workspace_id: Workspace ID

        Returns:
            Schema with relationship_types, memory_subtypes, and can_customize flag

        Example:
            schema = await client.get_workspace_schema("ws_123")
            print(schema["relationship_types"])
        """
        return await self._request("GET", f"/workspaces/{workspace_id}/schema")

    # Memory extension operations

    async def decay(
        self,
        memory_id: str,
        decay_rate: float = 0.1,
    ) -> Memory:
        """
        Reduce memory importance by decay rate.

        Args:
            memory_id: Memory ID
            decay_rate: Rate of decay 0.0-1.0 (default: 0.1)

        Returns:
            Updated memory with decayed importance

        Example:
            memory = await client.decay("mem_123", decay_rate=0.2)
        """
        payload = {"decay_rate": decay_rate}
        data = await self._request("POST", f"/memories/{memory_id}/decay", json=payload)
        return Memory(**data.get("memory", data))

    async def trace_memory(self, memory_id: str) -> dict[str, Any]:
        """
        Trace memory provenance back to source.

        Returns information about the memory's origin including
        source resource, category membership, and association chain.

        Args:
            memory_id: Memory ID

        Returns:
            Trace result with provenance information

        Example:
            trace = await client.trace_memory("mem_123")
            print(trace["chain"])
        """
        data = await self._request("GET", f"/memories/{memory_id}/trace")
        return data.get("trace", data)

    async def batch_memories(
        self,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform multiple memory operations in a single request.

        Supported operation types:
        - create: {"type": "create", "data": {"content": "...", ...}}
        - update: {"type": "update", "data": {"memory_id": "...", "content": "..."}}
        - delete: {"type": "delete", "data": {"memory_id": "...", "hard": False}}

        Args:
            operations: List of operations to perform

        Returns:
            Results for each operation with success/error status

        Example:
            results = await client.batch_memories([
                {"type": "create", "data": {"content": "Memory 1"}},
                {"type": "create", "data": {"content": "Memory 2"}},
                {"type": "delete", "data": {"memory_id": "mem_old"}}
            ])
        """
        payload = {"operations": operations}
        return await self._request("POST", "/memories/batch", json=payload)

    # Session extension operations

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its context data.

        Args:
            session_id: Session ID

        Returns:
            True if successful

        Example:
            await client.delete_session("sess_123")
        """
        await self._request("DELETE", f"/sessions/{session_id}")
        return True

    async def commit_session(
        self,
        session_id: str,
        min_importance: float = 0.5,
        deduplicate: bool = True,
        categories: Optional[list[str]] = None,
        max_memories: int = 50,
    ) -> dict[str, Any]:
        """
        Commit session working memory to long-term memory.

        Extracts memories from session working memory, deduplicates them,
        and creates new long-term memories.

        Args:
            session_id: Session ID
            min_importance: Minimum importance threshold (default: 0.5)
            deduplicate: Whether to deduplicate memories (default: True)
            categories: Filter by categories (optional)
            max_memories: Maximum memories to extract (default: 50)

        Returns:
            Commit result with extraction statistics

        Example:
            result = await client.commit_session("sess_123")
            print(f"Created {result['memories_created']} memories")
        """
        payload: dict[str, Any] = {
            "min_importance": min_importance,
            "deduplicate": deduplicate,
            "max_memories": max_memories,
        }
        if categories is not None:
            payload["categories"] = categories

        return await self._request("POST", f"/sessions/{session_id}/commit", json=payload)

    async def touch_session(self, session_id: str) -> dict[str, Any]:
        """
        Update session expiration (extend TTL).

        Args:
            session_id: Session ID

        Returns:
            Updated expiration timestamp

        Example:
            result = await client.touch_session("sess_123")
            print(f"Expires at: {result['expires_at']}")
        """
        return await self._request("POST", f"/sessions/{session_id}/touch")

    # Context Environment operations

    async def context_exec(
        self,
        code: str,
        result_var: Optional[str] = None,
        return_result: bool = True,
        max_return_chars: int = 10_000,
    ) -> dict[str, Any]:
        """
        Execute Python code in the session's sandbox environment.

        Requires an active session (call set_session() first).

        Args:
            code: Python code to execute
            result_var: If set, store the expression result in this variable
            return_result: Whether to include the result value in response
            max_return_chars: Maximum characters for result serialization

        Returns:
            Dict with keys: output, result, error, variables_changed

        Example:
            await client.context_exec("x = [1, 2, 3]")
            result = await client.context_exec("sum(x)")
            print(result["result"])  # 6
        """
        payload: dict[str, Any] = {
            "code": code,
            "return_result": return_result,
            "max_return_chars": max_return_chars,
        }
        if result_var is not None:
            payload["result_var"] = result_var

        return await self._request("POST", "/context/execute", json=payload)

    async def context_inspect(
        self,
        variable: Optional[str] = None,
        preview_chars: int = 200,
    ) -> dict[str, Any]:
        """
        Inspect sandbox state or a specific variable.

        Requires an active session (call set_session() first).

        Args:
            variable: Specific variable name to inspect (all if None)
            preview_chars: Maximum characters per variable preview

        Returns:
            Dict with variable names, types, and preview values

        Example:
            state = await client.context_inspect()
            print(state["variables"])
        """
        params: dict[str, Any] = {"preview_chars": preview_chars}
        if variable is not None:
            params["variable"] = variable

        return await self._request("POST", "/context/inspect", params=params)

    async def context_load(
        self,
        var: str,
        query: str,
        limit: int = 50,
        types: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        min_relevance: Optional[float] = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """
        Load memories into the sandbox as a variable.

        Runs a memory recall query and stores the results as a list
        of dicts in the specified variable.

        Requires an active session (call set_session() first).

        Args:
            var: Variable name to store results in
            query: Memory recall query
            limit: Maximum memories to recall
            types: Filter by memory types
            tags: Filter by tags
            min_relevance: Minimum relevance score
            include_embeddings: Whether to include embedding vectors

        Returns:
            Dict with count and variable info

        Example:
            result = await client.context_load("memories", "coding preferences")
            print(f"Loaded {result['count']} memories")
        """
        payload: dict[str, Any] = {
            "var": var,
            "query": query,
            "limit": limit,
            "include_embeddings": include_embeddings,
        }
        if types is not None:
            payload["types"] = types
        if tags is not None:
            payload["tags"] = tags
        if min_relevance is not None:
            payload["min_relevance"] = min_relevance

        return await self._request("POST", "/context/load", json=payload)

    async def context_inject(
        self,
        key: str,
        value: Any,
        parse_json: bool = False,
    ) -> dict[str, Any]:
        """
        Inject a value into the sandbox state.

        Requires an active session (call set_session() first).

        Args:
            key: Variable name
            value: Value to inject
            parse_json: If True, parse value as JSON string

        Returns:
            Dict confirming the injection

        Example:
            await client.context_inject("config", {"debug": True})
        """
        payload: dict[str, Any] = {
            "key": key,
            "value": value,
            "parse_json": parse_json,
        }

        return await self._request("POST", "/context/inject", json=payload)

    async def context_query(
        self,
        prompt: str,
        variables: list[str],
        max_context_chars: Optional[int] = None,
        result_var: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send sandbox variables and a prompt to the LLM.

        Requires an active session (call set_session() first).

        Args:
            prompt: User prompt for the LLM
            variables: Variable names to include as context
            max_context_chars: Maximum characters for variable context
            result_var: If set, store the LLM response in this variable

        Returns:
            Dict with the LLM response text and token usage

        Example:
            result = await client.context_query(
                "Summarize the data",
                variables=["memories", "stats"]
            )
            print(result["response"])
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "variables": variables,
        }
        if max_context_chars is not None:
            payload["max_context_chars"] = max_context_chars
        if result_var is not None:
            payload["result_var"] = result_var

        return await self._request("POST", "/context/query", json=payload)

    async def context_rlm(
        self,
        goal: str,
        memory_query: Optional[str] = None,
        memory_limit: int = 100,
        max_iterations: int = 10,
        variables: Optional[list[str]] = None,
        result_var: Optional[str] = None,
        detail_level: str = "standard",
    ) -> dict[str, Any]:
        """
        Run a Recursive Language Model (RLM) loop.

        Iteratively executes code and LLM queries to achieve a goal.

        Requires an active session (call set_session() first).

        Args:
            goal: Natural language description of the goal
            memory_query: Optional memory query to load initial data
            memory_limit: Maximum memories to load
            max_iterations: Maximum reasoning iterations
            variables: Variable names to include in context
            result_var: If set, store the final result in this variable
            detail_level: Level of detail: "brief", "standard", "detailed"

        Returns:
            Dict with result, iterations performed, and execution trace

        Example:
            result = await client.context_rlm(
                goal="Analyze user preferences and find contradictions",
                memory_query="user preferences",
            )
            print(result["result"])
        """
        payload: dict[str, Any] = {
            "goal": goal,
            "memory_limit": memory_limit,
            "max_iterations": max_iterations,
            "detail_level": detail_level,
        }
        if memory_query is not None:
            payload["memory_query"] = memory_query
        if variables is not None:
            payload["variables"] = variables
        if result_var is not None:
            payload["result_var"] = result_var

        return await self._request("POST", "/context/rlm", json=payload)

    async def context_status(self) -> dict[str, Any]:
        """
        Get the status of the session's sandbox environment.

        Requires an active session (call set_session() first).

        Returns:
            Dict with variable count, memory usage, and metadata

        Example:
            status = await client.context_status()
            print(f"Variables: {status['variable_count']}")
        """
        return await self._request("GET", "/context/status")

    async def context_checkpoint(self) -> None:
        """
        Checkpoint the session's sandbox state for persistence.

        Fires persistence hooks so enterprise deployments can save sandbox
        state to durable storage. No-op for default in-memory deployments.

        Requires an active session (call set_session() first).

        Example:
            await client.context_checkpoint()
        """
        await self._request("POST", "/context/checkpoint")

    async def context_cleanup(self) -> None:
        """
        Clean up and remove the session's sandbox environment.

        Requires an active session (call set_session() first).

        Example:
            await client.context_cleanup()
        """
        await self._request("DELETE", "/context/cleanup")
