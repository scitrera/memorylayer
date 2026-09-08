package memorylayer

import "context"

// PrincipalRef is a reference to a principal (user, service, agent, task).
type PrincipalRef struct {
	Type string
	ID   string
}

// AuthorityContext is an on-behalf-of (OBO) authority. The client emits it as
// X-Aether-* headers; the HTTP transport forwards them and the Aether transport
// translates them into the proxy's structured authorization envelope.
//
// A nil Subject means direct authority (acting as the grant holder); a non-nil
// Subject means acting on behalf of that principal.
type AuthorityContext struct {
	GrantID string
	Subject *PrincipalRef
}

// oboHeaders builds the X-Aether-* OBO headers for an authority, falling back to
// the client's default authority. Returns nil when neither is set.
func (c *Client) oboHeaders(authority *AuthorityContext) map[string]string {
	effective := authority
	if effective == nil {
		effective = c.defaultAuthority
	}
	if effective == nil {
		return nil
	}
	h := map[string]string{"X-Aether-Grant-ID": effective.GrantID}
	if effective.Subject != nil {
		h["X-Aether-Authority-Mode"] = "on_behalf_of"
		h["X-Aether-Subject-Type"] = effective.Subject.Type
		h["X-Aether-Subject-ID"] = effective.Subject.ID
	} else {
		h["X-Aether-Authority-Mode"] = "direct"
	}
	return h
}

// resolveAuthority returns the explicit authority, or the proxy's bound one.
func resolveAuthority(explicit, bound *AuthorityContext) *AuthorityContext {
	if explicit != nil {
		return explicit
	}
	return bound
}

// OBOProxy is a lightweight view over a [Client] that injects a fixed OBO
// authority and/or default workspace ID into every call, without mutating the
// shared client. Obtain one with [Client.ActingFor] or [Client.ForWorkspace].
//
// It mirrors the most common operations; for the full surface use the parent
// Client directly with per-call Authority/WorkspaceID options.
type OBOProxy struct {
	parent      *Client
	authority   *AuthorityContext
	workspaceID string
}

// ActingFor returns a proxy that carries OBO authority for the given grant and
// optional subject. A nil subject acts as the grant holder (direct mode).
//
//	alice := client.ActingFor("g_abc", &memorylayer.PrincipalRef{Type: "user", ID: "alice"})
//	alice.Recall(ctx, "preferences", memorylayer.RecallOptions{WorkspaceID: "ws_work"})
func (c *Client) ActingFor(grantID string, subject *PrincipalRef) *OBOProxy {
	return &OBOProxy{
		parent:      c,
		authority:   &AuthorityContext{GrantID: grantID, Subject: subject},
		workspaceID: c.workspaceID,
	}
}

// ForWorkspace returns a proxy with workspaceID baked in (using the client's
// default authority).
func (c *Client) ForWorkspace(workspaceID string) *OBOProxy {
	return &OBOProxy{parent: c, authority: c.defaultAuthority, workspaceID: workspaceID}
}

// ActingFor returns a new proxy with a different authority (workspace inherited).
func (p *OBOProxy) ActingFor(grantID string, subject *PrincipalRef) *OBOProxy {
	return &OBOProxy{
		parent:      p.parent,
		authority:   &AuthorityContext{GrantID: grantID, Subject: subject},
		workspaceID: p.workspaceID,
	}
}

// ForWorkspace returns a new proxy with a different workspace (authority
// inherited).
func (p *OBOProxy) ForWorkspace(workspaceID string) *OBOProxy {
	return &OBOProxy{parent: p.parent, authority: p.authority, workspaceID: workspaceID}
}

func (p *OBOProxy) ws(workspaceID string) string {
	if workspaceID != "" {
		return workspaceID
	}
	return p.workspaceID
}

// Remember stores a memory under the proxy's authority + workspace.
func (p *OBOProxy) Remember(ctx context.Context, content string, opts RememberOptions) (*Memory, error) {
	opts.WorkspaceID = p.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, p.authority)
	return p.parent.Remember(ctx, content, opts)
}

// Recall searches memories under the proxy's authority + workspace.
func (p *OBOProxy) Recall(ctx context.Context, query string, opts RecallOptions) (*RecallResult, error) {
	opts.WorkspaceID = p.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, p.authority)
	return p.parent.Recall(ctx, query, opts)
}

// Reflect synthesizes memories under the proxy's authority.
func (p *OBOProxy) Reflect(ctx context.Context, query string, opts ReflectOptions) (*ReflectResult, error) {
	opts.Authority = resolveAuthority(opts.Authority, p.authority)
	return p.parent.Reflect(ctx, query, opts)
}

// AppendMessages appends messages under the proxy's authority + workspace.
func (p *OBOProxy) AppendMessages(ctx context.Context, threadID string, messages []map[string]any, opts AppendMessagesOptions) ([]ChatMessage, error) {
	opts.WorkspaceID = p.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, p.authority)
	return p.parent.AppendMessages(ctx, threadID, messages, opts)
}

// DeleteMessage deletes a message under the proxy's authority + workspace.
func (p *OBOProxy) DeleteMessage(ctx context.Context, threadID, messageID string, opts DeleteMessageOptions) (bool, error) {
	opts.WorkspaceID = p.ws(opts.WorkspaceID)
	opts.Authority = resolveAuthority(opts.Authority, p.authority)
	return p.parent.DeleteMessage(ctx, threadID, messageID, opts)
}
