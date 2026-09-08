package memorylayer

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Request is a single MemoryLayer API request as seen by a [Transport]. Path is
// relative to the API version root (e.g. "/memories", without the "/v1"
// prefix); transports are responsible for mounting it under "/v1". Body is the
// already-encoded payload (JSON or multipart); Header carries the content-type,
// auth, session, and X-Aether-* OBO headers the Client assembled.
type Request struct {
	Method string
	Path   string
	Query  url.Values
	Header http.Header
	Body   []byte
}

// Response is the subset of an HTTP response the Client's request layer relies
// on. Body is fully buffered.
type Response struct {
	StatusCode int
	Header     http.Header
	Body       []byte
}

// Transport issues a single MemoryLayer API request. Implementations:
//
//   - The default HTTP transport wraps an *http.Client and talks directly to a
//     MemoryLayer server.
//   - The Aether transport (companion module) routes via the Aether SDK's
//     proxy_http against the MemoryLayer service topic.
//
// A Transport must be safe for concurrent use.
type Transport interface {
	// RoundTrip issues req and returns the buffered response. It returns a
	// non-nil error only for transport-level failures (the Client maps HTTP
	// status codes to typed errors itself).
	RoundTrip(ctx context.Context, req *Request) (*Response, error)
	// Close releases any resources the transport owns. Transports that wrap a
	// caller-owned connection (e.g. Aether) leave it open.
	Close() error
}

// Methods safe to retry automatically. POST is conditionally retryable only
// when the caller supplied an Idempotency-Key.
var idempotentMethods = map[string]bool{
	http.MethodGet:    true,
	http.MethodHead:   true,
	http.MethodPut:    true,
	http.MethodDelete: true,
	http.MethodPatch:  true,
	"OPTIONS":         true,
}

// HTTP status codes considered transient and worth retrying.
var retryableStatus = map[int]bool{
	429: true, 500: true, 502: true, 503: true, 504: true,
}

// httpTransport is the default Client transport: one *http.Client talking
// directly to a MemoryLayer server. It adds bounded retry-with-backoff for
// idempotent requests on transient failures, honoring Retry-After.
type httpTransport struct {
	baseURL    string // already suffixed with the version prefix ("/v1")
	httpClient *http.Client
	maxRetries int
	backoff    time.Duration
}

func newHTTPTransport(baseURL string, hc *http.Client, timeout time.Duration, maxRetries int) *httpTransport {
	if hc == nil {
		hc = &http.Client{Timeout: timeout}
	}
	if maxRetries < 0 {
		maxRetries = 0
	}
	return &httpTransport{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: hc,
		maxRetries: maxRetries,
		backoff:    500 * time.Millisecond,
	}
}

func (t *httpTransport) RoundTrip(ctx context.Context, req *Request) (*Response, error) {
	u := t.baseURL + req.Path
	if len(req.Query) > 0 {
		u += "?" + req.Query.Encode()
	}

	method := strings.ToUpper(req.Method)
	retryable := (idempotentMethods[method] || (method == http.MethodPost && req.Header.Get("Idempotency-Key") != "")) && t.maxRetries > 0

	for attempt := 0; ; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, req.Method, u, bodyReader(req.Body))
		if err != nil {
			return nil, err
		}
		copyHeader(httpReq.Header, req.Header)

		resp, err := t.httpClient.Do(httpReq)
		if err != nil {
			if !retryable || attempt >= t.maxRetries {
				return nil, err
			}
			if werr := sleepBackoff(ctx, t.backoff, attempt, ""); werr != nil {
				return nil, werr
			}
			continue
		}

		body, readErr := io.ReadAll(resp.Body)
		// The body is already drained above; a Close error here tells us nothing
		// actionable and must not mask readErr.
		_ = resp.Body.Close()
		if readErr != nil {
			if !retryable || attempt >= t.maxRetries {
				return nil, readErr
			}
			if werr := sleepBackoff(ctx, t.backoff, attempt, ""); werr != nil {
				return nil, werr
			}
			continue
		}

		if retryable && retryableStatus[resp.StatusCode] && attempt < t.maxRetries {
			if werr := sleepBackoff(ctx, t.backoff, attempt, resp.Header.Get("Retry-After")); werr != nil {
				return nil, werr
			}
			continue
		}

		return &Response{StatusCode: resp.StatusCode, Header: resp.Header, Body: body}, nil
	}
}

func (t *httpTransport) Close() error {
	t.httpClient.CloseIdleConnections()
	return nil
}

func bodyReader(b []byte) io.Reader {
	if len(b) == 0 {
		return nil
	}
	return bytes.NewReader(b)
}

func copyHeader(dst, src http.Header) {
	for k, vs := range src {
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

// sleepBackoff waits before the next retry, honoring a numeric Retry-After
// (seconds) when present, otherwise exponential backoff. It returns ctx.Err()
// if the context is canceled while waiting.
func sleepBackoff(ctx context.Context, base time.Duration, attempt int, retryAfter string) error {
	delay := base * time.Duration(1<<attempt)
	if retryAfter != "" {
		if secs, err := strconv.ParseFloat(retryAfter, 64); err == nil {
			if ra := time.Duration(secs * float64(time.Second)); ra > delay {
				delay = ra
			}
		}
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
