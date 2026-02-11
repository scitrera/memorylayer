"""Custom exceptions for MemoryLayer.ai SDK."""


class MemoryLayerError(Exception):
    """Base exception for all MemoryLayer errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(MemoryLayerError):
    """Raised when authentication fails (401)."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class AuthorizationError(MemoryLayerError):
    """Raised when authorization is denied (403)."""

    def __init__(self, message: str = "Authorization denied") -> None:
        super().__init__(message, status_code=403)


class NotFoundError(MemoryLayerError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class ValidationError(MemoryLayerError):
    """Raised when request validation fails (422)."""

    def __init__(self, message: str = "Validation error") -> None:
        super().__init__(message, status_code=422)


class RateLimitError(MemoryLayerError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)


class ServerError(MemoryLayerError):
    """Raised when server returns 5xx error."""

    def __init__(self, message: str = "Server error", status_code: int = 500) -> None:
        super().__init__(message, status_code=status_code)
