package aether

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/url"
	"testing"

	aethersdk "github.com/scitrera/aether/sdk/go/aether"
	memorylayer "github.com/scitrera/memorylayer/memorylayer-sdk-go"
)

// fakeProxy records the request it receives and returns a canned response.
type fakeProxy struct {
	gotTarget string
	gotReq    *http.Request
	gotBody   []byte
	resp      *http.Response
}

func (f *fakeProxy) ProxyHTTP(ctx context.Context, target string, req *http.Request, opts ...aethersdk.ProxyOpt) (*http.Response, error) {
	f.gotTarget = target
	f.gotReq = req
	if req.Body != nil {
		f.gotBody, _ = io.ReadAll(req.Body)
	}
	if f.resp != nil {
		return f.resp, nil
	}
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewReader([]byte(`{"ok":true}`))),
	}, nil
}

func TestTransportMountsPathAndQuery(t *testing.T) {
	fp := &fakeProxy{}
	tr := NewTransport(fp, WithTarget("sv::memorylayer::test"))

	req := &memorylayer.Request{
		Method: http.MethodPost,
		Path:   "/memories/recall",
		Query:  url.Values{"a": {"1"}},
		Header: http.Header{"Content-Type": {"application/json"}},
		Body:   []byte(`{"query":"x"}`),
	}
	resp, err := tr.RoundTrip(context.Background(), req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	if fp.gotTarget != "sv::memorylayer::test" {
		t.Errorf("target = %q", fp.gotTarget)
	}
	if got := fp.gotReq.URL.RequestURI(); got != "/v1/memories/recall?a=1" {
		t.Errorf("request URI = %q, want /v1/memories/recall?a=1", got)
	}
	if string(fp.gotBody) != `{"query":"x"}` {
		t.Errorf("body = %q", fp.gotBody)
	}
	if resp.StatusCode != 200 || string(resp.Body) != `{"ok":true}` {
		t.Errorf("response = %d %q", resp.StatusCode, resp.Body)
	}
}

func TestTransportStripsOBOHeaders(t *testing.T) {
	fp := &fakeProxy{}
	tr := NewTransport(fp)

	req := &memorylayer.Request{
		Method: http.MethodGet,
		Path:   "/memories",
		Header: http.Header{
			"Authorization":           {"Bearer k"},
			"X-Aether-Grant-Id":       {"g_1"},
			"X-Aether-Authority-Mode": {"on_behalf_of"},
			"X-Aether-Subject-Type":   {"user"},
			"X-Aether-Subject-Id":     {"alice"},
		},
	}
	if _, err := tr.RoundTrip(context.Background(), req); err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	// OBO headers must be extracted into the proxy envelope, not forwarded as
	// HTTP headers (strict-mode terminators strip them anyway).
	for _, h := range []string{"X-Aether-Grant-Id", "X-Aether-Authority-Mode", "X-Aether-Subject-Type", "X-Aether-Subject-Id"} {
		if fp.gotReq.Header.Get(h) != "" {
			t.Errorf("header %q should have been stripped", h)
		}
	}
	// Non-OBO headers pass through.
	if fp.gotReq.Header.Get("Authorization") != "Bearer k" {
		t.Errorf("Authorization header was dropped")
	}
}

func TestTransportDefaultTarget(t *testing.T) {
	fp := &fakeProxy{}
	tr := NewTransport(fp)
	if _, err := tr.RoundTrip(context.Background(), &memorylayer.Request{Method: http.MethodGet, Path: "/healthz"}); err != nil {
		t.Fatalf("RoundTrip: %v", err)
	}
	if fp.gotTarget != DefaultTarget {
		t.Errorf("target = %q, want %q", fp.gotTarget, DefaultTarget)
	}
}
