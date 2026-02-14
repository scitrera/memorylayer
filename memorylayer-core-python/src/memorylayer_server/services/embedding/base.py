import base64
from logging import Logger

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from scitrera_app_framework.api import Variables, Plugin, enabled_option_pattern
from scitrera_app_framework import get_extension, get_logger, ext_parse_bool

from ...config import (
    MEMORYLAYER_EMBEDDING_PROVIDER, DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER,
    MEMORYLAYER_EMBEDDING_SERVICE, DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE,
    MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED, DEFAULT_MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED
)
from .._constants import EXT_CACHE_SERVICE, EXT_EMBEDDING_PROVIDER, EXT_EMBEDDING_SERVICE


class EmbeddingType(str, Enum):
    """Type of content being embedded."""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"  # Combined text + image


@dataclass
class EmbeddingInput:
    """Input for embedding generation, supporting multimodal content."""
    text: Optional[str] = None
    image: Optional[Union[str, bytes, Path]] = None  # Base64, bytes, URL, or file path

    def __post_init__(self):
        if not self.text and not self.image:
            raise ValueError("At least one of text or image must be provided")

    @property
    def embedding_type(self) -> EmbeddingType:
        if self.text and self.image:
            return EmbeddingType.MULTIMODAL
        elif self.image:
            return EmbeddingType.IMAGE
        return EmbeddingType.TEXT

    def to_dict(self) -> dict:
        """Convert to dict format for Qwen3-VL."""
        result = {}
        if self.text:
            result["text"] = self.text
        if self.image:
            result["image"] = self.image if isinstance(self.image, str) else None
        return result


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    def __init__(self, v: Variables = None, output_dimensions: Optional[int] = None):
        self._dimensions = output_dimensions
        self.logger = get_logger(v, name=self.__class__.__name__)

    async def preload(self):
        """
        Preload any required resources.

        This is optional (not all embedding providers need to implement it, but can improve time to first embedding
        for large models.)
        """
        return

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        pass

    @property
    def dimensions(self) -> int:
        """Embedding dimensions."""
        return self._dimensions


class MultimodalEmbeddingProvider(EmbeddingProvider):
    """
    Abstract multimodal embedding provider that supports text, images, and combined content.

    Extends base EmbeddingProvider with multimodal capabilities.
    """

    @abstractmethod
    async def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Generate embedding for an image."""
        pass

    @abstractmethod
    async def embed_multimodal(
            self,
            text: Optional[str] = None,
            image: Optional[Union[str, bytes, Path]] = None
    ) -> list[float]:
        """Generate embedding for combined text and image."""
        pass

    async def embed_input(self, input: EmbeddingInput) -> list[float]:
        """Generate embedding for EmbeddingInput (convenience method)."""
        if input.embedding_type == EmbeddingType.TEXT:
            return await self.embed(input.text)
        elif input.embedding_type == EmbeddingType.IMAGE:
            return await self.embed_image(input.image)
        else:
            return await self.embed_multimodal(input.text, input.image)

    @staticmethod
    def load_image_bytes(image: Union[str, bytes, Path]) -> bytes:
        """Load image as bytes from various input formats."""
        if isinstance(image, bytes):
            return image
        elif isinstance(image, Path):
            return image.read_bytes()
        elif isinstance(image, str):
            # Check if it's base64 or a file path
            if image.startswith("data:image"):
                # Data URL format
                _, encoded = image.split(",", 1)
                return base64.b64decode(encoded)
            elif image.startswith(("http://", "https://")):
                # URL - download
                import urllib.request
                with urllib.request.urlopen(image) as response:
                    return response.read()
            elif len(image) > 500 and not Path(image).exists():
                # Likely base64 string
                return base64.b64decode(image)
            else:
                # File path
                return Path(image).read_bytes()
        raise ValueError(f"Unsupported image type: {type(image)}")


# noinspection PyAbstractClass
class EmbeddingProviderPluginBase(Plugin):
    """Base Plugin Implementation for embedding providers."""
    PROVIDER_NAME: str = ''

    def name(self) -> str:
        return f"{EXT_EMBEDDING_PROVIDER}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_EMBEDDING_PROVIDER

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_EMBEDDING_PROVIDER, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_EMBEDDING_PROVIDER, DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER)

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        # noinspection PyTypeChecker
        embedding_provider: EmbeddingProvider = value
        preload = v.environ(MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED,
                            default=DEFAULT_MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED, type_fn=ext_parse_bool)

        # initiate preload if implemented in the embedding provider
        if preload:
            embedding_provider.logger.info("Attempting to preload embedding model in background")
            await embedding_provider.preload()
        return


# noinspection PyAbstractClass
class EmbeddingServicePluginBase(Plugin):
    """Base plugin for embedding service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_EMBEDDING_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_EMBEDDING_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_EMBEDDING_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_EMBEDDING_SERVICE, DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_EMBEDDING_PROVIDER, EXT_CACHE_SERVICE)
