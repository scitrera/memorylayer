export class MemoryLayerError extends Error {
  constructor(message: string, public statusCode?: number, public details?: unknown) {
    super(message);
    this.name = "MemoryLayerError";
  }
}

export class AuthenticationError extends MemoryLayerError {
  constructor(message = "Authentication failed") {
    super(message, 401);
    this.name = "AuthenticationError";
  }
}

export class AuthorizationError extends MemoryLayerError {
  constructor(message = "Authorization denied") {
    super(message, 403);
    this.name = "AuthorizationError";
  }
}

export class NotFoundError extends MemoryLayerError {
  constructor(message = "Resource not found") {
    super(message, 404);
    this.name = "NotFoundError";
  }
}

export class ValidationError extends MemoryLayerError {
  constructor(message = "Validation failed", details?: unknown) {
    super(message, 400, details);
    this.name = "ValidationError";
  }
}

export class RateLimitError extends MemoryLayerError {
  constructor(message = "Rate limit exceeded", public retryAfter?: number) {
    super(message, 429);
    this.name = "RateLimitError";
  }
}
