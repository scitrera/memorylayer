"""Synchronous client wrapper for MemoryLayer.ai SDK."""

import json
import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

import httpx
from pydantic import TypeAdapter

from .exceptions import (
    AuthenticationError,
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


class SyncMemoryLayerClient:
    """
    Synchronous Python client for MemoryLayer.ai API.

    Usage:
        with SyncMemoryLayerClient(
            base_url="https://api.memorylayer.ai",
            api_key="your-api-key",
            workspace_id="ws_123"
        ) as client:
            # Store a memory
            memory = client.remember(
                content="User prefers Python",
                type=MemoryType.SEMANTIC,
                importance=0.8
            )

            # Search memories
            results = client.recall("coding preferences")

            # Reflect on memories
            reflection = client.reflect("summarize user preferences")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:61001",
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize MemoryLayer client.

        Args:
            base_url: API base URL (default: http://localhost:61001)
            api_key: API key for authentication
            workspace_id: Default workspace ID for operations
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "SyncMemoryLayerClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Initialize the HTTP client."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.workspace_id:
            headers["X-Workspace-ID"] = self.workspace_id

        self._client = httpx.Client(
            base_url=f"{self.base_url}/v1",
            headers=headers,
            timeout=self.timeout,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()

    def _ensure_client(self) -> httpx.Client:
        """Ensure client is initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use context manager or call connect().")
        return self._client

    def _request(
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
            response = client.request(method, path, json=json, params=params)

            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError(response.json().get("detail", "Authentication failed"))
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

    def remember(
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
            type: Cognitive memory type (episodic, semantic, procedural, working).
                Accepts MemoryType enum or plain string.
            subtype: Domain subtype (Solution, Problem, etc.).
                Accepts MemorySubtype enum or plain string.
            importance: Importance score 0.0-1.0 (default: 0.5)
            tags: Tags for categorization
            metadata: Additional metadata
            space_id: Optional memory space ID

        Returns:
            Created memory

        Example:
            memory = client.remember(
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

        data = self._request("POST", "/memories", json=payload)
        return Memory(**data)

    def recall(
        self,
        query: str,
        types: Optional[list[Union[str, MemoryType]]] = None,
        subtypes: Optional[list[Union[str, MemorySubtype]]] = None,
        tags: Optional[list[str]] = None,
        mode: Optional[Union[str, RecallMode]] = None,
        limit: int = 10,
        min_relevance: Optional[float] = None,
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
            tolerance: Search tolerance (loose, moderate, strict). None = server default.
            include_associations: Include linked memories (None = server default)
            traverse_depth: Multi-hop graph traversal depth (None = server default)
            max_expansion: Max memories discovered via graph expansion (None = server default)

        Returns:
            Recall results with memories

        Example:
            results = client.recall(
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
        if max_expansion is not None:
            payload["max_expansion"] = max_expansion
        if types:
            payload["types"] = [_to_value(t) for t in types]
        if subtypes:
            payload["subtypes"] = [_to_value(s) for s in subtypes]
        if tags:
            payload["tags"] = tags

        data = self._request("POST", "/memories/recall", json=payload)

        # Parse memories
        memories_adapter = TypeAdapter(list[Memory])
        memories = memories_adapter.validate_python(data.get("memories", []))

        return RecallResult(
            memories=memories,
            total_count=data.get("total_count", len(memories)),
            query_tokens=data.get("query_tokens"),
            search_latency_ms=data.get("search_latency_ms"),
        )

    def reflect(
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
            reflection = client.reflect(
                query="summarize everything about user's development workflow",
                max_tokens=300
            )
        """
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "include_sources": include_sources,
        }

        data = self._request("POST", "/memories/reflect", json=payload)
        return ReflectResult(**data)

    def forget(
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
            client.forget("mem_123", hard=False)
        """
        params = {"hard": "true" if hard else "false"}
        self._request("DELETE", f"/memories/{memory_id}", params=params)
        return True

    def get_memory(self, memory_id: str) -> Memory:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object

        Example:
            memory = client.get_memory("mem_123")
        """
        data = self._request("GET", f"/memories/{memory_id}")
        return Memory(**data)

    def update_memory(
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
            memory = client.update_memory(
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

        data = self._request("PATCH", f"/memories/{memory_id}", json=payload)
        return Memory(**data)

    # Association methods

    def associate(
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
            assoc = client.associate(
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

        data = self._request("POST", f"/memories/{source_id}/associate", json=payload)
        return Association(**data)

    def get_associations(
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
            associations = client.get_associations("mem_123")
        """
        params = {"direction": direction}
        data = self._request("GET", f"/memories/{memory_id}/associations", params=params)

        associations_adapter = TypeAdapter(list[Association])
        return associations_adapter.validate_python(data.get("associations", []))

    # Session methods

    def create_session(self, ttl_seconds: int = 3600) -> Session:
        """
        Create a new working memory session.

        Args:
            ttl_seconds: Time to live in seconds (default: 3600 = 1 hour)

        Returns:
            Created session

        Example:
            session = client.create_session(ttl_seconds=7200)
        """
        payload = {"ttl_seconds": ttl_seconds}
        data = self._request("POST", "/sessions", json=payload)
        return Session(**data)

    def get_session(self, session_id: str) -> Session:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object
        """
        data = self._request("GET", f"/sessions/{session_id}")
        return Session(**data)

    def set_context(
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
            client.set_context(
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

        self._request("POST", f"/sessions/{session_id}/context", json=payload)

    def get_context(
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
            context = client.get_context(
                "sess_123",
                ["current_file", "user_intent"]
            )
        """
        params = {"keys": ",".join(keys)}
        data = self._request("GET", f"/sessions/{session_id}/context", params=params)
        return data.get("context", {})

    def get_briefing(
        self,
        lookback_hours: Optional[int] = None,
        lookback_minutes: int = 60,
        detail_level: str = "abstract",
        limit: int = 10,
        include_memories: bool = True,
        include_contradictions: bool = True,
    ) -> SessionBriefing:
        """
        Get a session briefing with recent activity and context.

        Args:
            lookback_hours: DEPRECATED - Use lookback_minutes instead. If provided, overrides lookback_minutes.
            lookback_minutes: Time window in minutes for recent memories (default: 60)
            detail_level: Level of memory detail - "abstract", "overview", or "full" (default: "abstract")
            limit: Maximum number of recent memories to include (default: 10)
            include_memories: Whether to include recent memory content (default: True)
            include_contradictions: Flag contradicting memories in briefing (default: True)

        Returns:
            Session briefing

        Example:
            briefing = client.get_briefing(lookback_minutes=120, detail_level="full")
        """
        params = {
            "lookback_minutes": lookback_hours * 60 if lookback_hours is not None else lookback_minutes,
            "detail_level": detail_level,
            "limit": limit,
            "include_memories": str(include_memories).lower(),
            "include_contradictions": str(include_contradictions).lower(),
        }
        data = self._request("GET", "/sessions/briefing", params=params)
        return SessionBriefing(**data)

    # Workspace methods

    def create_workspace(self, name: str) -> Workspace:
        """
        Create a new workspace.

        Args:
            name: Workspace name

        Returns:
            Created workspace

        Example:
            workspace = client.create_workspace("my-project")
        """
        payload = {"name": name}
        data = self._request("POST", "/workspaces", json=payload)
        return Workspace(**data)

    def get_workspace(self, workspace_id: Optional[str] = None) -> Workspace:
        """
        Get workspace details.

        Args:
            workspace_id: Workspace ID (uses default if not provided)

        Returns:
            Workspace object

        Example:
            workspace = client.get_workspace()
        """
        ws_id = workspace_id or self.workspace_id
        if not ws_id:
            raise ValueError("workspace_id must be provided or set on client")

        data = self._request("GET", f"/workspaces/{ws_id}")
        return Workspace(**data)

    # Context Environment operations

    def context_exec(
        self,
        code: str,
        result_var: Optional[str] = None,
        return_result: bool = True,
        max_return_chars: int = 10_000,
    ) -> dict[str, Any]:
        """Execute Python code in the session's sandbox environment."""
        payload: dict[str, Any] = {
            "code": code,
            "return_result": return_result,
            "max_return_chars": max_return_chars,
        }
        if result_var is not None:
            payload["result_var"] = result_var
        return self._request("POST", "/context/execute", json=payload)

    def context_inspect(
        self,
        variable: Optional[str] = None,
        preview_chars: int = 200,
    ) -> dict[str, Any]:
        """Inspect sandbox state or a specific variable."""
        params: dict[str, Any] = {"preview_chars": preview_chars}
        if variable is not None:
            params["variable"] = variable
        return self._request("POST", "/context/inspect", params=params)

    def context_load(
        self,
        var: str,
        query: str,
        limit: int = 50,
        types: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        min_relevance: Optional[float] = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Load memories into the sandbox as a variable."""
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
        return self._request("POST", "/context/load", json=payload)

    def context_inject(
        self,
        key: str,
        value: Any,
        parse_json: bool = False,
    ) -> dict[str, Any]:
        """Inject a value into the sandbox state."""
        payload: dict[str, Any] = {
            "key": key,
            "value": value,
            "parse_json": parse_json,
        }
        return self._request("POST", "/context/inject", json=payload)

    def context_query(
        self,
        prompt: str,
        variables: list[str],
        max_context_chars: Optional[int] = None,
        result_var: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send sandbox variables and a prompt to the LLM."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "variables": variables,
        }
        if max_context_chars is not None:
            payload["max_context_chars"] = max_context_chars
        if result_var is not None:
            payload["result_var"] = result_var
        return self._request("POST", "/context/query", json=payload)

    def context_rlm(
        self,
        goal: str,
        memory_query: Optional[str] = None,
        memory_limit: int = 100,
        max_iterations: int = 10,
        variables: Optional[list[str]] = None,
        result_var: Optional[str] = None,
        detail_level: str = "standard",
    ) -> dict[str, Any]:
        """Run a Recursive Language Model (RLM) loop."""
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
        return self._request("POST", "/context/rlm", json=payload)

    def context_status(self) -> dict[str, Any]:
        """Get the status of the session's sandbox environment."""
        return self._request("GET", "/context/status")

    def context_checkpoint(self) -> None:
        """Checkpoint the session's sandbox state for persistence."""
        self._request("POST", "/context/checkpoint")

    def context_cleanup(self) -> None:
        """Clean up and remove the session's sandbox environment."""
        self._request("DELETE", "/context/cleanup")

    def export_workspace(
        self,
        workspace_id: Optional[str] = None,
        include_associations: bool = True,
        offset: int = 0,
        limit: int = 0,
    ) -> dict:
        """Export workspace memories and associations as JSON.

        Args:
            workspace_id: Workspace to export (uses default if not specified)
            include_associations: Whether to include associations
            offset: Skip first N memories (default: 0)
            limit: Maximum memories to export (default: 0 = all)

        Returns:
            Export data dictionary with memories and associations

        Example:
            data = client.export_workspace("ws_123")
            print(f"Exported {len(data['memories'])} memories")
        """
        ws_id = workspace_id or self.workspace_id
        if not ws_id:
            raise ValueError("Workspace ID required")
        params = {
            "include_associations": str(include_associations).lower(),
            "offset": str(offset),
            "limit": str(limit),
        }

        # Fetch raw NDJSON response
        client = self._ensure_client()
        response = client.get(f"/workspaces/{ws_id}/export", params=params)

        # Handle errors
        if response.status_code >= 400:
            if response.status_code == 401:
                raise AuthenticationError(response.json().get("detail", "Authentication failed"))
            elif response.status_code == 404:
                raise NotFoundError(response.json().get("detail", "Resource not found"))
            elif response.status_code == 422:
                raise ValidationError(response.json().get("detail", "Validation error"))
            else:
                raise MemoryLayerError(
                    response.json().get("detail", "Request failed"),
                    status_code=response.status_code,
                )

        response.raise_for_status()

        # Parse NDJSON response
        text = response.text
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        header = None
        memories = []
        associations = []

        for line in lines:
            parsed = json.loads(line)
            if parsed.get("type") == "header":
                header = parsed
            elif parsed.get("type") == "memory":
                memories.append(parsed["data"])
            elif parsed.get("type") == "association":
                associations.append(parsed["data"])

        # Build backward-compatible response
        return {
            "version": header.get("version", "1.0") if header else "1.0",
            "workspace_id": header.get("workspace_id", ws_id) if header else ws_id,
            "exported_at": header.get("exported_at", "") if header else "",
            "total_memories": header.get("total_memories", len(memories)) if header else len(memories),
            "total_associations": header.get("total_associations", len(associations)) if header else len(associations),
            "memories": memories,
            "associations": associations,
        }

    def import_workspace(
        self,
        workspace_id: str,
        data: dict,
    ) -> dict:
        """Import memories and associations into a workspace.

        Args:
            workspace_id: Target workspace ID
            data: Export data dictionary (from export_workspace)

        Returns:
            Import results with counts of imported/skipped/errors

        Example:
            result = client.import_workspace("ws_123", export_data)
            print(f"Imported {result['imported']} memories")
        """
        response = self._request(
            "POST", f"/workspaces/{workspace_id}/import",
            json={"data": data}
        )
        return response

    def export_workspace_stream(
        self,
        workspace_id: Optional[str] = None,
        include_associations: bool = True,
        offset: int = 0,
        limit: int = 0,
    ) -> Generator[dict, None, None]:
        """Export workspace as streaming NDJSON.

        Yields parsed JSON objects line-by-line (header, memory, association, footer).

        Args:
            workspace_id: Workspace to export (uses default if not specified)
            include_associations: Whether to include associations
            offset: Skip first N memories (default: 0)
            limit: Maximum memories to export (default: 0 = all)

        Yields:
            Parsed JSON objects from NDJSON stream

        Example:
            for line in client.export_workspace_stream("ws_123"):
                if line["type"] == "memory":
                    print(f"Memory: {line['data']['content']}")
        """
        ws_id = workspace_id or self.workspace_id
        if not ws_id:
            raise ValueError("Workspace ID required")
        params = {
            "include_associations": str(include_associations).lower(),
            "offset": str(offset),
            "limit": str(limit),
        }

        client = self._ensure_client()
        response = client.get(f"/workspaces/{ws_id}/export", params=params)

        if response.status_code >= 400:
            if response.status_code == 401:
                raise AuthenticationError(response.json().get("detail", "Authentication failed"))
            elif response.status_code == 404:
                raise NotFoundError(response.json().get("detail", "Resource not found"))
            elif response.status_code == 422:
                raise ValidationError(response.json().get("detail", "Validation error"))
            else:
                raise MemoryLayerError(
                    response.json().get("detail", "Request failed"),
                    status_code=response.status_code,
                )

        response.raise_for_status()

        text = response.text
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        for line in lines:
            yield json.loads(line)

    def import_workspace_stream(
        self,
        workspace_id: str,
        ndjson_lines: list[dict],
    ) -> dict:
        """Import workspace from NDJSON format.

        Args:
            workspace_id: Target workspace ID
            ndjson_lines: List of dicts to serialize as NDJSON

        Returns:
            Import results with counts of imported/skipped/errors

        Example:
            lines = [
                {"type": "header", "version": "1.0", ...},
                {"type": "memory", "data": {...}},
            ]
            result = client.import_workspace_stream("ws_123", lines)
            print(f"Imported {result['imported']} memories")
        """
        # Serialize to NDJSON
        ndjson_body = '\n'.join(json.dumps(line) for line in ndjson_lines)

        client = self._ensure_client()
        response = client.post(
            f"/workspaces/{workspace_id}/import",
            content=ndjson_body,
            headers={"Content-Type": "application/x-ndjson"}
        )

        if response.status_code >= 400:
            if response.status_code == 401:
                raise AuthenticationError(response.json().get("detail", "Authentication failed"))
            elif response.status_code == 404:
                raise NotFoundError(response.json().get("detail", "Resource not found"))
            elif response.status_code == 422:
                raise ValidationError(response.json().get("detail", "Validation error"))
            else:
                raise MemoryLayerError(
                    response.json().get("detail", "Request failed"),
                    status_code=response.status_code,
                )

        response.raise_for_status()
        return response.json()


@contextmanager
def sync_client(
    base_url: str = "http://localhost:61001",
    api_key: Optional[str] = None,
    workspace_id: Optional[str] = None,
    timeout: float = 30.0,
) -> Generator[SyncMemoryLayerClient, None, None]:
    """
    Context manager for synchronous MemoryLayer client.

    Args:
        base_url: API base URL (default: http://localhost:61001)
        api_key: API key for authentication
        workspace_id: Default workspace ID for operations
        timeout: Request timeout in seconds (default: 30.0)

    Yields:
        SyncMemoryLayerClient instance

    Example:
        with sync_client(api_key="key", workspace_id="ws_123") as client:
            memory = client.remember("User prefers Python")
            results = client.recall("coding preferences")
    """
    client = SyncMemoryLayerClient(
        base_url=base_url,
        api_key=api_key,
        workspace_id=workspace_id,
        timeout=timeout,
    )
    with client:
        yield client
