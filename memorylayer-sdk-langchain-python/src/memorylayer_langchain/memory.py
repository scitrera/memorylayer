"""Legacy BaseMemory implementations using MemoryLayer as the backend."""

import logging
from typing import Any

import httpx
from langchain_classic.base_memory import BaseMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ConfigDict

from memorylayer import SyncMemoryLayerClient

logger = logging.getLogger(__name__)

# Default summarization prompt template
DEFAULT_SUMMARY_PROMPT = (
    "Provide a concise summary of the conversation history for session {session_id}. "
    "Focus on key topics, decisions, and important context that would be useful "
    "for continuing the conversation."
)


class MemoryLayerMemory(BaseMemory):
    """
    LangChain BaseMemory implementation backed by MemoryLayer.

    This class provides a drop-in replacement for LangChain's
    ConversationBufferMemory, storing conversation history in MemoryLayer
    for persistence across sessions.

    Note: This class is for legacy LangChain chains. For LCEL-based chains,
    use MemoryLayerChatMessageHistory with RunnableWithMessageHistory instead.

    Usage:
        from memorylayer_langchain import MemoryLayerMemory

        # Create memory instance
        memory = MemoryLayerMemory(
            session_id="user_123_conversation_1",
            base_url="http://localhost:61001",
            api_key="your-api-key",
            workspace_id="ws_123"
        )

        # Use with legacy LangChain chains
        from langchain.chains import ConversationChain
        chain = ConversationChain(llm=llm, memory=memory)
        chain.run("Hello!")

    Attributes:
        session_id: Unique identifier for this conversation session.
        memory_key: Key used for memory variables (default: "history").
        human_prefix: Prefix for human messages (default: "Human").
        ai_prefix: Prefix for AI messages (default: "AI").
        return_messages: If True, returns list of messages instead of string.
        max_messages: Maximum number of messages to retrieve (default: 1000).
    """

    # Class-level configuration - these define the fields for this Pydantic model
    session_id: str
    base_url: str = "http://localhost:61001"
    api_key: str | None = None
    workspace_id: str | None = None
    timeout: float = 30.0
    memory_tags: list[str] = []

    # Memory formatting options
    memory_key: str = "history"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    input_key: str | None = None
    output_key: str | None = None
    return_messages: bool = False
    max_messages: int = 1000  # Maximum number of messages to retrieve

    # Private state
    _headers: dict[str, str] = {}
    _message_count: int = 0
    _client: SyncMemoryLayerClient | None = None

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Initialize MemoryLayerMemory with configuration."""
        super().__init__(**data)

        # Build headers for API requests (kept for backward compatibility with tests)
        self._headers = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        if self.workspace_id:
            self._headers["X-Workspace-ID"] = self.workspace_id

        # Initialize the synchronous client
        self._client = SyncMemoryLayerClient(
            base_url=self.base_url,
            api_key=self.api_key,
            workspace_id=self.workspace_id,
            timeout=self.timeout,
        )
        self._client.connect()

        # Initialize message counter
        self._message_count = self._get_current_message_count()

    def close(self) -> None:
        """Close the underlying MemoryLayer client and release resources."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_method()
        except Exception:
            # Avoid raising during cleanup; just log the exception.
            logger.exception("Error while closing SyncMemoryLayerClient")

    def __enter__(self) -> "MemoryLayerMemory":
        """Allow use as a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Ensure resources are cleaned up when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if the instance is garbage-collected."""
        try:
            self.close()
        except Exception:
            # Never propagate exceptions from __del__.
            pass

    def _get_current_message_count(self) -> int:
        """Get the current message count for this session."""
        try:
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"conversation history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "conversation_memory"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )
            return len(result.memories)
        except httpx.HTTPStatusError:
            return 0

    def _get_memories(self) -> list[Any]:
        """Retrieve all memories for this session, ordered by index."""
        try:
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"conversation history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "conversation_memory"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )

            memories = result.memories

            # Sort by message_index from metadata
            memories.sort(key=lambda m: m.metadata.get("message_index", 0))

            return memories

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    @property
    def memory_variables(self) -> list[str]:
        """
        Return list of memory variable names.

        Returns:
            List containing the memory_key (default: ["history"])
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Load memory variables for the chain.

        This retrieves the conversation history from MemoryLayer and
        returns it formatted according to configuration.

        Args:
            inputs: Current chain inputs (not used for this implementation)

        Returns:
            Dictionary with memory_key mapped to conversation history.
            If return_messages is True, returns list of BaseMessage objects.
            Otherwise, returns formatted string.
        """
        memories = self._get_memories()

        if self.return_messages:
            # Return as list of LangChain messages
            messages = []
            for memory in memories:
                role = memory.metadata.get("role", "human")
                content = memory.content

                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                else:
                    # Default to AIMessage for any other roles to preserve
                    # existing behavior for unexpected/legacy role values.
                    messages.append(AIMessage(content=content))

            return {self.memory_key: messages}
        else:
            # Return as formatted string
            buffer_string = ""
            for memory in memories:
                role = memory.metadata.get("role", "human")
                content = memory.content

                if role == "human":
                    buffer_string += f"{self.human_prefix}: {content}\n"
                elif role == "ai":
                    buffer_string += f"{self.ai_prefix}: {content}\n"
                elif role == "system":
                    # System messages are formatted like AI messages
                    buffer_string += f"{self.ai_prefix}: {content}\n"
                else:
                    # For any other unexpected roles, use ai_prefix
                    buffer_string += f"{self.ai_prefix}: {content}\n"

            return {self.memory_key: buffer_string.strip()}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """
        Save the context of a conversation turn to memory.

        This stores both the human input and AI output as separate
        memories in MemoryLayer, enabling retrieval of the full
        conversation history.

        Args:
            inputs: Dictionary containing the human input.
                If input_key is set, uses that key.
                Otherwise, uses the first key in the dict.
            outputs: Dictionary containing the AI output.
                If output_key is set, uses that key.
                Otherwise, uses the first key in the dict.
        """
        # Extract input
        if self.input_key:
            input_str = inputs.get(self.input_key, "")
        else:
            input_str = next(iter(inputs.values()), "") if inputs else ""

        # Extract output
        if self.output_key:
            output_str = outputs.get(self.output_key, "")
        else:
            output_str = next(iter(outputs.values()), "") if outputs else ""

        # Ensure strings
        if not isinstance(input_str, str):
            input_str = str(input_str)
        if not isinstance(output_str, str):
            output_str = str(output_str)

        # Store human message
        self._store_memory(
            content=input_str,
            role="human",
            message_index=self._message_count,
        )
        self._message_count += 1

        # Store AI message
        self._store_memory(
            content=output_str,
            role="ai",
            message_index=self._message_count,
        )
        self._message_count += 1

    def _store_memory(
        self,
        content: str,
        role: str,
        message_index: int,
    ) -> None:
        """
        Store a single memory in MemoryLayer.

        Args:
            content: The message content
            role: Either "human" or "ai"
            message_index: Index for ordering messages
        """
        tags = [
            f"session:{self.session_id}",
            "conversation_memory",
            f"role:{role}",
            *self.memory_tags,
        ]

        try:
            from memorylayer import MemoryType
            self._client.remember(
                content=content,
                type=MemoryType.EPISODIC,
                importance=0.5,
                tags=tags,
                metadata={
                    "session_id": self.session_id,
                    "role": role,
                    "message_index": message_index,
                },
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def clear(self) -> None:
        """
        Clear all memories for this session.

        This deletes all memories tagged with this session's ID from
        MemoryLayer.
        """
        try:
            memories = self._get_memories()

            for memory in memories:
                try:
                    self._client.forget(memory.id)
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Failed to delete memory {memory.id}: {e}")

            # Reset message counter
            self._message_count = 0

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to clear memories: {e}")
            raise


class MemoryLayerConversationSummaryMemory(BaseMemory):
    """
    LangChain BaseMemory implementation that returns a summary of the conversation.

    This class uses MemoryLayer's reflect endpoint to generate a concise
    summary of the conversation history rather than returning the full
    conversation transcript. This is useful for keeping context windows
    small while maintaining important conversation context.

    Note: This class is for legacy LangChain chains. For LCEL-based chains,
    use MemoryLayerChatMessageHistory with RunnableWithMessageHistory instead.

    Usage:
        from memorylayer_langchain import MemoryLayerConversationSummaryMemory

        # Create memory instance
        memory = MemoryLayerConversationSummaryMemory(
            session_id="user_123_conversation_1",
            base_url="http://localhost:61001",
            api_key="your-api-key",
            workspace_id="ws_123"
        )

        # Use with legacy LangChain chains
        from langchain.chains import ConversationChain
        chain = ConversationChain(llm=llm, memory=memory)
        chain.run("Hello!")

    Attributes:
        session_id: Unique identifier for this conversation session.
        memory_key: Key used for memory variables (default: "history").
        summary_prompt: Custom prompt template for summarization.
        max_tokens: Maximum tokens for the summary (default: 500).
        return_messages: If True, returns summary as a SystemMessage.
        max_messages: Maximum number of messages to retrieve (default: 1000).
    """

    # Class-level configuration - these define the fields for this Pydantic model
    session_id: str
    base_url: str = "http://localhost:61001"
    api_key: str | None = None
    workspace_id: str | None = None
    timeout: float = 30.0
    memory_tags: list[str] = []

    # Memory formatting options
    memory_key: str = "history"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    input_key: str | None = None
    output_key: str | None = None
    return_messages: bool = False

    # Summary-specific options
    summary_prompt: str | None = None
    max_tokens: int = 500
    include_sources: bool = False
    max_messages: int = 1000  # Maximum number of messages to retrieve

    # Private state
    _headers: dict[str, str] = {}
    _message_count: int = 0
    _client: SyncMemoryLayerClient | None = None

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Initialize MemoryLayerConversationSummaryMemory with configuration."""
        super().__init__(**data)

        # Build headers for API requests (kept for backward compatibility with tests)
        self._headers = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        if self.workspace_id:
            self._headers["X-Workspace-ID"] = self.workspace_id

        # Initialize the synchronous client
        self._client = SyncMemoryLayerClient(
            base_url=self.base_url,
            api_key=self.api_key,
            workspace_id=self.workspace_id,
            timeout=self.timeout,
        )
        self._client.connect()

        # Initialize message counter
        self._message_count = self._get_current_message_count()

    def close(self) -> None:
        """Close the underlying MemoryLayer client and release resources."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_method()
        except Exception:
            # Avoid raising during cleanup; just log the exception.
            logger.exception("Error while closing SyncMemoryLayerClient")

    def __enter__(self) -> "MemoryLayerConversationSummaryMemory":
        """Allow use as a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Ensure resources are cleaned up when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if the instance is garbage-collected."""
        try:
            self.close()
        except Exception:
            # Never propagate exceptions from __del__.
            pass

    def _get_current_message_count(self) -> int:
        """Get the current message count for this session."""
        try:
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"conversation history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "conversation_memory"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )
            return len(result.memories)
        except httpx.HTTPStatusError:
            return 0

    def _get_summary_query(self) -> str:
        """Get the summarization query to use for reflection."""
        if self.summary_prompt:
            return self.summary_prompt.format(session_id=self.session_id)
        return DEFAULT_SUMMARY_PROMPT.format(session_id=self.session_id)

    def _get_reflection(self) -> str:
        """
        Get a summary of the conversation using the reflect endpoint.

        Returns:
            Summary of the conversation, or empty string if no memories.
        """
        try:
            result = self._client.reflect(
                query=self._get_summary_query(),
                max_tokens=self.max_tokens,
                include_sources=self.include_sources,
            )

            # Extract the reflection from the response (note: SDK uses 'reflection', not 'synthesis')
            return result.reflection

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get reflection: {e}")
            return ""

    @property
    def memory_variables(self) -> list[str]:
        """
        Return list of memory variable names.

        Returns:
            List containing the memory_key (default: ["history"])
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Load memory variables for the chain.

        This retrieves a summary of the conversation history from MemoryLayer
        using the reflect endpoint and returns it formatted according to
        configuration.

        Args:
            inputs: Current chain inputs (not used for this implementation)

        Returns:
            Dictionary with memory_key mapped to conversation summary.
            If return_messages is True, returns list with a SystemMessage.
            Otherwise, returns the summary string.
        """
        summary = self._get_reflection()

        if self.return_messages:
            # Return as a list with a single SystemMessage containing the summary
            if summary:
                return {self.memory_key: [SystemMessage(content=summary)]}
            return {self.memory_key: []}
        else:
            # Return as formatted string
            return {self.memory_key: summary}

    def save_context(
        self, inputs: dict[str, Any], outputs: dict[str, str]
    ) -> None:
        """
        Save the context of a conversation turn to memory.

        This stores both the human input and AI output as separate
        memories in MemoryLayer, enabling the reflect endpoint to
        synthesize them into a summary.

        Args:
            inputs: Dictionary containing the human input.
                If input_key is set, uses that key.
                Otherwise, uses the first key in the dict.
            outputs: Dictionary containing the AI output.
                If output_key is set, uses that key.
                Otherwise, uses the first key in the dict.
        """
        # Extract input
        if self.input_key:
            input_str = inputs.get(self.input_key, "")
        else:
            input_str = next(iter(inputs.values()), "") if inputs else ""

        # Extract output
        if self.output_key:
            output_str = outputs.get(self.output_key, "")
        else:
            output_str = next(iter(outputs.values()), "") if outputs else ""

        # Ensure strings
        if not isinstance(input_str, str):
            input_str = str(input_str)
        if not isinstance(output_str, str):
            output_str = str(output_str)

        # Store human message
        self._store_memory(
            content=input_str,
            role="human",
            message_index=self._message_count,
        )
        self._message_count += 1

        # Store AI message
        self._store_memory(
            content=output_str,
            role="ai",
            message_index=self._message_count,
        )
        self._message_count += 1

    def _store_memory(
        self,
        content: str,
        role: str,
        message_index: int,
    ) -> None:
        """
        Store a single memory in MemoryLayer.

        Args:
            content: The message content
            role: Either "human" or "ai"
            message_index: Index for ordering messages
        """
        tags = [
            f"session:{self.session_id}",
            "conversation_memory",
            f"role:{role}",
            *self.memory_tags,
        ]

        try:
            from memorylayer import MemoryType
            self._client.remember(
                content=content,
                type=MemoryType.EPISODIC,
                importance=0.5,
                tags=tags,
                metadata={
                    "session_id": self.session_id,
                    "role": role,
                    "message_index": message_index,
                },
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    def _get_memories(self) -> list[Any]:
        """Retrieve all memories for this session, ordered by index."""
        try:
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"conversation history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "conversation_memory"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )

            memories = result.memories

            # Sort by message_index from metadata
            memories.sort(key=lambda m: m.metadata.get("message_index", 0))

            return memories

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def clear(self) -> None:
        """
        Clear all memories for this session.

        This deletes all memories tagged with this session's ID from
        MemoryLayer.
        """
        try:
            memories = self._get_memories()

            for memory in memories:
                try:
                    self._client.forget(memory.id)
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Failed to delete memory {memory.id}: {e}")

            # Reset message counter
            self._message_count = 0

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to clear memories: {e}")
            raise
