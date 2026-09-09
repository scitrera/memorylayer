// Package aether provides a MemoryLayer SDK [memorylayer.Transport] that routes
// every request through an existing Aether SDK connection via proxy_http,
// instead of opening a direct HTTP path to the MemoryLayer server.
//
// It is a separate Go module so that HTTP-only callers of the core
// github.com/scitrera/memorylayer/memorylayer-sdk-go package do not pull in the
// Aether SDK's gRPC dependency tree.
//
// # Usage
//
//	agentClient, _ := aethersdk.NewAgentClient(aethersdk.AgentOptions{ /* ... */ })
//	// ... agentClient.Connect(ctx) ...
//
//	tr := aether.NewTransport(agentClient, aether.WithTarget("sv::memorylayer"))
//	client, _ := memorylayer.NewClient(
//		memorylayer.WithTransport(tr),
//		memorylayer.WithAPIKey("..."),
//		memorylayer.WithWorkspaceID("ws_123"),
//	)
//
// The transport does NOT own the Aether connection: [Transport.Close] is a
// no-op, and disposing of the underlying client is the caller's responsibility.
package aether

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	pb "github.com/scitrera/aether/api/proto"
	aethersdk "github.com/scitrera/aether/sdk/go/aether"
	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
)

// DefaultTarget is the Aether service topic used when none is configured. The
// bare form lets Aether route to an available memorylayer service instance.
const DefaultTarget = "sv::memorylayer"

// ProxyClient is the subset of the Aether SDK client the transport needs. Both
// *aethersdk.AgentClient and *aethersdk.ServiceClient satisfy it (each embeds
// *aethersdk.BaseClient, which provides ProxyHTTP).
type ProxyClient interface {
	ProxyHTTP(ctx context.Context, target string, req *http.Request, opts ...aethersdk.ProxyOpt) (*http.Response, error)
}

var _ memorylayer.Transport = (*Transport)(nil)

// Transport implements [memorylayer.Transport] over an Aether connection.
type Transport struct {
	client   ProxyClient
	target   string
	basePath string
	backend  string
	timeout  time.Duration
}

// Option configures a [Transport].
type Option func(*Transport)

// WithTarget sets the Aether service topic (default [DefaultTarget]). Use an
// explicit "sv::memorylayer::<specifier>" to pin a specific instance.
func WithTarget(target string) Option {
	return func(t *Transport) { t.target = target }
}

// WithBackend pins requests to a named terminator backend (its allow-list still
// applies).
func WithBackend(backend string) Option {
	return func(t *Transport) { t.backend = backend }
}

// WithTimeout sets a per-request timeout, applied as a context deadline when the
// caller's context has none (default 30s).
func WithTimeout(d time.Duration) Option {
	return func(t *Transport) { t.timeout = d }
}

// WithBasePath overrides the API version path the SDK is mounted under (default
// "/v1"), matching the terminator's allow_paths.
func WithBasePath(basePath string) Option {
	return func(t *Transport) { t.basePath = strings.TrimRight(basePath, "/") }
}

// NewTransport builds an Aether transport over an already-connected Aether
// client (an *aethersdk.AgentClient or *aethersdk.ServiceClient).
func NewTransport(client ProxyClient, opts ...Option) *Transport {
	t := &Transport{
		client:   client,
		target:   DefaultTarget,
		basePath: "/v1",
		timeout:  30 * time.Second,
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// OBO headers the core SDK emits; the transport intercepts these and converts
// them into the proxy's structured AuthorizationContext envelope (strict-mode
// terminators strip inbound X-Aether-* headers, so the proto field is the only
// reliable OBO channel).
const (
	hdrGrantID     = "X-Aether-Grant-Id"
	hdrMode        = "X-Aether-Authority-Mode"
	hdrSubjectType = "X-Aether-Subject-Type"
	hdrSubjectID   = "X-Aether-Subject-Id"
)

// RoundTrip implements [memorylayer.Transport].
func (t *Transport) RoundTrip(ctx context.Context, req *memorylayer.Request) (*memorylayer.Response, error) {
	fullPath := req.Path
	if !strings.HasPrefix(fullPath, "/") {
		fullPath = "/" + fullPath
	}
	fullPath = t.basePath + fullPath

	// Build the request URL (host is ignored by the proxy; only the request URI
	// — path + query — is forwarded).
	target := "http://memorylayer" + fullPath
	if len(req.Query) > 0 {
		target += "?" + req.Query.Encode()
	}

	var body io.Reader
	if len(req.Body) > 0 {
		body = bytes.NewReader(req.Body)
	}
	httpReq, err := http.NewRequest(req.Method, target, body)
	if err != nil {
		return nil, err
	}

	// Copy headers, extracting OBO headers into a structured envelope.
	auth := &pb.AuthorizationContext{}
	for k, vs := range req.Header {
		switch http.CanonicalHeaderKey(k) {
		case hdrGrantID:
			auth.GrantId = firstValue(vs)
		case hdrMode:
			auth.AuthorityMode = firstValue(vs)
		case hdrSubjectType:
			ensureSubject(auth).PrincipalType = firstValue(vs)
		case hdrSubjectID:
			ensureSubject(auth).PrincipalId = firstValue(vs)
		default:
			for _, v := range vs {
				httpReq.Header.Add(k, v)
			}
		}
	}

	// Apply a per-request timeout when the caller's context has no deadline.
	if t.timeout > 0 {
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, t.timeout)
			defer cancel()
		}
	}

	if auth.GetGrantId() != "" {
		if auth.GetAuthorityMode() == "" {
			auth.AuthorityMode = "direct"
		}
		ctx = aethersdk.WithOBOAuthorization(ctx, auth)
	}

	var opts []aethersdk.ProxyOpt
	if t.backend != "" {
		opts = append(opts, aethersdk.WithBackend(t.backend))
	}

	// The aether SDK's ProxyHTTP reads the OBO AuthorizationContext from the
	// REQUEST's context (req.Context()), not from the ctx argument — the ctx arg
	// is consulted only for the deadline. So the OBO set on ctx above must be
	// carried on httpReq, otherwise it is dropped and the proxy call goes out in
	// direct mode as the bare connection identity (the relay sidecar service),
	// losing the user's OBO grant → MemoryLayer denies on-behalf-of writes.
	httpReq = httpReq.WithContext(ctx)
	resp, err := t.client.ProxyHTTP(ctx, t.target, httpReq, opts...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &memorylayer.Response{
		StatusCode: resp.StatusCode,
		Header:     resp.Header,
		Body:       respBody,
	}, nil
}

// Close is a no-op: the transport does not own the Aether connection.
func (t *Transport) Close() error { return nil }

func ensureSubject(auth *pb.AuthorizationContext) *pb.PrincipalRef {
	if auth.Subject == nil {
		auth.Subject = &pb.PrincipalRef{}
	}
	return auth.Subject
}

func firstValue(vs []string) string {
	if len(vs) == 0 {
		return ""
	}
	return vs[0]
}
