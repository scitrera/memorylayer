package memorylayer

import (
	"context"
	"net/http"
	"net/url"
)

// CreateTokenOptions configures [Client.CreateToken].
type CreateTokenOptions struct {
	PrincipalType     string   // default "User"
	WorkspacePatterns []string // default ["*"]
	Scopes            []string // default ["*"]
	ExpiresInDays     *int     // nil = no expiry
	Authority         *AuthorityContext
}

// CreateToken creates a new API token. The result includes the one-time
// plaintext Token secret — store it securely; it cannot be retrieved again.
func (c *Client) CreateToken(ctx context.Context, name string, opts CreateTokenOptions) (*TokenCreateResult, error) {
	principalType := opts.PrincipalType
	if principalType == "" {
		principalType = "User"
	}
	patterns := opts.WorkspacePatterns
	if patterns == nil {
		patterns = []string{"*"}
	}
	scopes := opts.Scopes
	if scopes == nil {
		scopes = []string{"*"}
	}
	payload := map[string]any{
		"name":               name,
		"principal_type":     principalType,
		"workspace_patterns": patterns,
		"scopes":             scopes,
	}
	if opts.ExpiresInDays != nil {
		payload["expires_in_days"] = *opts.ExpiresInDays
	}
	var out TokenCreateResult
	if err := c.requestJSON(ctx, http.MethodPost, "/tokens", payload,
		requestSpec{authority: opts.Authority}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListTokens lists API tokens (without plaintext secrets).
func (c *Client) ListTokens(ctx context.Context, includeRevoked bool, authority *AuthorityContext) ([]TokenInfo, error) {
	q := url.Values{"include_revoked": {boolStr(includeRevoked)}}
	var out struct {
		Tokens []TokenInfo `json:"tokens"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/tokens",
		query:     q,
		authority: authority,
	}, &out); err != nil {
		return nil, err
	}
	return out.Tokens, nil
}

// GetToken fetches a single API token by ID (without the plaintext secret).
func (c *Client) GetToken(ctx context.Context, tokenID string, authority *AuthorityContext) (*TokenInfo, error) {
	var out TokenInfo
	if err := c.doJSON(ctx, requestSpec{
		method:    http.MethodGet,
		path:      "/tokens/" + tokenID,
		authority: authority,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteToken deletes an API token.
func (c *Client) DeleteToken(ctx context.Context, tokenID string, authority *AuthorityContext) error {
	return c.doJSON(ctx, requestSpec{
		method:    http.MethodDelete,
		path:      "/tokens/" + tokenID,
		authority: authority,
	}, nil)
}

// RevokeToken revokes an API token. Revoked tokens are invalidated immediately
// but remain visible in listings with Revoked=true.
func (c *Client) RevokeToken(ctx context.Context, tokenID string, authority *AuthorityContext) error {
	return c.doJSON(ctx, requestSpec{
		method:    http.MethodPost,
		path:      "/tokens/" + tokenID + "/revoke",
		authority: authority,
	}, nil)
}
