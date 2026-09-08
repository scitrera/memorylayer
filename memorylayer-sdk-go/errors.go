package memorylayer

import (
	"errors"
	"fmt"
)

// APIError is the base error type for all MemoryLayer API failures. Every typed
// error below embeds an *APIError, so errors.As against *APIError matches any of
// them and exposes the HTTP StatusCode.
type APIError struct {
	Message    string
	StatusCode int
}

func (e *APIError) Error() string {
	if e.StatusCode != 0 {
		return fmt.Sprintf("memorylayer: %s (status %d)", e.Message, e.StatusCode)
	}
	return "memorylayer: " + e.Message
}

// AuthenticationError is returned for HTTP 401 responses.
type AuthenticationError struct{ *APIError }

// AuthorizationError is returned for HTTP 403 responses.
type AuthorizationError struct{ *APIError }

// NotFoundError is returned for HTTP 404 responses.
type NotFoundError struct{ *APIError }

// ConflictError is returned for HTTP 409 responses.
type ConflictError struct{ *APIError }

// PreconditionFailedError is returned for HTTP 412 stale/mismatched ETags.
type PreconditionFailedError struct{ *APIError }

// PreconditionRequiredError is returned for HTTP 428 responses when a required
// idempotency or conditional request header is absent.
type PreconditionRequiredError struct{ *APIError }

// EnterpriseRequiredError is returned when an Enterprise-only endpoint is hit on
// a server that does not provide the feature (HTTP 501 on a method flagged with
// an enterprise feature). The Feature field names the missing capability.
type EnterpriseRequiredError struct {
	*APIError
	Feature string
}

// ValidationError is returned for HTTP 422 responses.
type ValidationError struct{ *APIError }

// RateLimitError is returned for HTTP 429 responses.
type RateLimitError struct{ *APIError }

// ServerError is returned for HTTP 5xx responses.
type ServerError struct{ *APIError }

func newError(message string, statusCode int) *APIError {
	return &APIError{Message: message, StatusCode: statusCode}
}

// IsNotFound reports whether err is (or wraps) a NotFoundError.
func IsNotFound(err error) bool {
	var e *NotFoundError
	return errors.As(err, &e)
}

// IsEnterpriseRequired reports whether err is (or wraps) an
// EnterpriseRequiredError.
func IsEnterpriseRequired(err error) bool {
	var e *EnterpriseRequiredError
	return errors.As(err, &e)
}
