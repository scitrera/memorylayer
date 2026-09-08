package memorylayer

import (
	"bytes"
	"encoding/json"
	"time"
)

// Memory is a stored memory entry.
type Memory struct {
	ID                 string         `json:"id"`
	LogicalKey         *string        `json:"logical_key,omitempty"`
	WorkspaceID        string         `json:"workspace_id"`
	SpaceID            *string        `json:"space_id,omitempty"`
	UserID             *string        `json:"user_id,omitempty"`
	Content            string         `json:"content"`
	Type               MemoryType     `json:"type"`
	Subtype            *string        `json:"subtype,omitempty"`
	Importance         float64        `json:"importance"`
	Tags               []string       `json:"tags"`
	Metadata           map[string]any `json:"metadata"`
	RefinementMetadata map[string]any `json:"refinement_metadata"`
	AccessCount        int            `json:"access_count"`
	LastAccessedAt     *time.Time     `json:"last_accessed_at,omitempty"`
	CreatedAt          time.Time      `json:"created_at"`
	UpdatedAt          time.Time      `json:"updated_at"`

	// Additional server fields.
	ContentHash    *string    `json:"content_hash,omitempty"`
	ContextID      *string    `json:"context_id,omitempty"`
	TenantID       string     `json:"tenant_id"`
	DecayFactor    float64    `json:"decay_factor"`
	Pinned         bool       `json:"pinned"`
	Status         string     `json:"status"`
	Abstract       *string    `json:"abstract,omitempty"`
	Overview       *string    `json:"overview,omitempty"`
	SessionID      *string    `json:"session_id,omitempty"`
	SourceMemoryID *string    `json:"source_memory_id,omitempty"`
	ObserverID     *string    `json:"observer_id,omitempty"`
	SubjectID      *string    `json:"subject_id,omitempty"`
	RelevanceScore *float64   `json:"relevance_score,omitempty"`
	BoostedScore   *float64   `json:"boosted_score,omitempty"`
	SourceScope    *string    `json:"source_scope,omitempty"`
	Embedding      []float64  `json:"embedding,omitempty"`
	ArchivedAt     *time.Time `json:"archived_at,omitempty"`
	Revision       int        `json:"revision"`
	ETag           string     `json:"etag"`
	DeletedAt      *time.Time `json:"deleted_at,omitempty"`
	Relations      []map[string]any `json:"relations,omitempty"`
	RelationWriteResult *EntityRelationWriteResult `json:"relation_write_result,omitempty"`
}

// EntityRelationInput is an explicit typed entity relationship evidenced by a
// memory. Each endpoint may be supplied by canonical ID or exact/alias name.
type EntityRelationInput struct {
	SourceEntityID   string  `json:"source_entity_id,omitempty"`
	SourceEntityName string  `json:"source_entity_name,omitempty"`
	TargetEntityID   string  `json:"target_entity_id,omitempty"`
	TargetEntityName string  `json:"target_entity_name,omitempty"`
	Relationship     string  `json:"relationship"`
	Confidence       float64 `json:"confidence,omitempty"`
	SourceSpanStart  *int    `json:"source_span_start,omitempty"`
	SourceSpanEnd    *int    `json:"source_span_end,omitempty"`
	EvidenceKind     string  `json:"evidence_kind,omitempty"`
	ExtractionMethod string  `json:"extraction_method,omitempty"`
}

// EntityRelationWriteResult reports deterministic relation resolution and
// storage outcomes for a remember request.
type EntityRelationWriteResult struct {
	Resolved    int      `json:"resolved"`
	Unresolved  int      `json:"unresolved"`
	Rejected    int      `json:"rejected"`
	Duplicate   int      `json:"duplicate"`
	RelationIDs []string `json:"relation_ids"`
	Errors      []string `json:"errors"`
}

// EntityRelation is a canonical typed edge between two workspace entities.
type EntityRelation struct {
	ID             string    `json:"id"`
	WorkspaceID    string    `json:"workspace_id"`
	SourceEntityID string    `json:"source_entity_id"`
	TargetEntityID string    `json:"target_entity_id"`
	Relationship   string    `json:"relationship"`
	Direction      string    `json:"direction"`
	Confidence     float64   `json:"confidence"`
	Active         bool      `json:"active"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

// EntityRelationPath is a bounded relation traversal returned by recall.
type EntityRelationPath struct {
	SeedEntityID     string           `json:"seed_entity_id"`
	EntityIDs        []string         `json:"entity_ids"`
	Relations        []EntityRelation `json:"relations"`
	EvidenceIDs      []string         `json:"evidence_ids"`
	EvidenceMemoryIDs []string        `json:"evidence_memory_ids"`
}

// BudgetSummary describes deterministic token-budget enforcement.
type BudgetSummary struct {
	Requested      *int   `json:"requested,omitempty"`
	Used           int    `json:"used"`
	Estimator      string `json:"estimator"`
	TruncatedItems int    `json:"truncated_items"`
	OmittedItems   int    `json:"omitted_items"`
}

// GenerationSummary reports generation-boundary activity for an operation.
type GenerationSummary struct {
	Policy             string `json:"policy"`
	Calls              int    `json:"calls"`
	InputTokens        int    `json:"input_tokens"`
	OutputTokens       int    `json:"output_tokens"`
	DeniedCalls        int    `json:"denied_calls"`
	BudgetDeniedCalls  int    `json:"budget_denied_calls"`
}

// Association is a relationship (graph edge) between two memories.
type Association struct {
	ID           string         `json:"id"`
	WorkspaceID  string         `json:"workspace_id"`
	SourceID     string         `json:"source_id"`
	TargetID     string         `json:"target_id"`
	Relationship string         `json:"relationship"`
	Strength     float64        `json:"strength"`
	Metadata     map[string]any `json:"metadata"`
	CreatedAt    time.Time      `json:"created_at"`
}

// RecallResult is the result of a recall (search) or list query.
type RecallResult struct {
	Memories            []Memory             `json:"memories"`
	TotalCount          int                  `json:"total_count"`
	QueryTokens         *int                 `json:"query_tokens,omitempty"`
	SearchLatencyMs     *int                 `json:"search_latency_ms,omitempty"`
	RetrievalConfidence string               `json:"retrieval_confidence,omitempty"`
	ConfidenceReasons   []string             `json:"confidence_reasons,omitempty"`
	BudgetSummary       *BudgetSummary       `json:"budget_summary,omitempty"`
	GenerationSummary   *GenerationSummary   `json:"generation_summary,omitempty"`
	RelationPaths       []EntityRelationPath `json:"relation_paths,omitempty"`
}

// SessionCheckpoint is a durable raw pre-compaction transcript capture.
type SessionCheckpoint struct {
	ID               string    `json:"id"`
	WorkspaceID      string    `json:"workspace_id"`
	SessionID        string    `json:"session_id"`
	RawMemoryID      string    `json:"raw_memory_id"`
	SourceKind       string    `json:"source_kind"`
	SourceSequence   *int      `json:"source_sequence,omitempty"`
	SourceBoundary   *int      `json:"source_boundary,omitempty"`
	ContentHash      string    `json:"content_hash"`
	ByteCount        int       `json:"byte_count"`
	CaptureStatus    string    `json:"capture_status"`
	IndexStatus      string    `json:"index_status"`
	EnrichmentStatus string    `json:"enrichment_status"`
	IdempotencyKey   string    `json:"idempotency_key"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// ContextPackItem is one deterministic context-pack or delta item.
type ContextPackItem struct {
	ID               string     `json:"id"`
	Kind             string     `json:"kind"`
	Content          string     `json:"content"`
	Importance       float64    `json:"importance"`
	EventTime        *time.Time `json:"event_time,omitempty"`
	MatchSignals     []string   `json:"match_signals"`
	SourceReferences []string   `json:"source_references"`
	Tombstone        bool       `json:"tombstone"`
}

// ContextPack is a deterministic, hard-budgeted session resume payload.
type ContextPack struct {
	Rendered                 string           `json:"rendered"`
	Items                    []ContextPackItem `json:"items"`
	OpenThreads              []map[string]any `json:"open_threads"`
	UnresolvedContradictions []map[string]any `json:"unresolved_contradictions"`
	BudgetSummary            BudgetSummary    `json:"budget_summary"`
	GenerationSummary        GenerationSummary `json:"generation_summary"`
	Cursor                    string           `json:"cursor"`
	DegradationNotices       []string          `json:"degradation_notices"`
}

// ContextDelta is the changed session context since a context-pack cursor.
type ContextDelta struct {
	Rendered           string            `json:"rendered"`
	Items              []ContextPackItem `json:"items"`
	BudgetSummary      BudgetSummary     `json:"budget_summary"`
	GenerationSummary  GenerationSummary `json:"generation_summary"`
	Cursor             string            `json:"cursor"`
	HasMore            bool              `json:"has_more"`
	DegradationNotices []string          `json:"degradation_notices"`
}

// ReflectResult is the result of a reflect (synthesis) query.
type ReflectResult struct {
	Reflection      string   `json:"reflection"`
	SourceMemories  []string `json:"source_memories"`
	Confidence      float64  `json:"confidence"`
	TokensProcessed *int     `json:"tokens_processed,omitempty"`
}

// Session is a working-memory session.
type Session struct {
	ID          string         `json:"id"`
	WorkspaceID string         `json:"workspace_id"`
	UserID      *string        `json:"user_id,omitempty"`
	Metadata    map[string]any `json:"metadata"`
	ExpiresAt   time.Time      `json:"expires_at"`
	CreatedAt   time.Time      `json:"created_at"`
}

// SessionActivity summarizes recent activity within a session.
type SessionActivity struct {
	Timestamp       time.Time `json:"timestamp"`
	Summary         string    `json:"summary"`
	MemoriesCreated int       `json:"memories_created"`
	KeyDecisions    []string  `json:"key_decisions"`
}

// OpenThread is an open topic/thread of work surfaced in a briefing.
type OpenThread struct {
	Topic        string    `json:"topic"`
	Status       string    `json:"status"`
	LastActivity time.Time `json:"last_activity"`
	KeyMemories  []string  `json:"key_memories"`
}

// ContradictionDetection is a detected contradiction between two memories.
type ContradictionDetection struct {
	MemoryA         string `json:"memory_a"`
	MemoryB         string `json:"memory_b"`
	Relationship    string `json:"relationship"`
	NeedsResolution bool   `json:"needs_resolution"`
}

// SessionBriefing is a session briefing with recent activity and context.
type SessionBriefing struct {
	WorkspaceSummary       map[string]any           `json:"workspace_summary"`
	RecentActivity         []SessionActivity        `json:"recent_activity"`
	OpenThreads            []OpenThread             `json:"open_threads"`
	ContradictionsDetected []ContradictionDetection `json:"contradictions_detected"`
}

// Workspace is a workspace (tenant boundary).
type Workspace struct {
	ID        string         `json:"id"`
	TenantID  string         `json:"tenant_id"`
	Name      string         `json:"name"`
	Settings  map[string]any `json:"settings"`
	Tags      []string       `json:"tags"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

// Context is a logical grouping of memories within a workspace.
type Context struct {
	ID          string         `json:"id"`
	WorkspaceID string         `json:"workspace_id"`
	Name        string         `json:"name"`
	Description *string        `json:"description,omitempty"`
	Settings    map[string]any `json:"settings,omitempty"`
	CreatedAt   *time.Time     `json:"created_at,omitempty"`
}

// TokenInfo is an API token record (without the plaintext secret).
type TokenInfo struct {
	ID                string   `json:"id"`
	Name              string   `json:"name"`
	PrincipalType     string   `json:"principal_type"`
	WorkspacePatterns []string `json:"workspace_patterns"`
	Scopes            []string `json:"scopes"`
	CreatedAt         string   `json:"created_at"`
	ExpiresAt         *string  `json:"expires_at,omitempty"`
	Revoked           bool     `json:"revoked"`
}

// TokenCreateResult is a newly created token, including the one-time plaintext
// secret. The Token is only returned at creation time — store it securely.
type TokenCreateResult struct {
	TokenInfo
	Token string `json:"token"`
}

// Entity is a canonical, workspace-scoped entity node.
type Entity struct {
	ID                     string         `json:"id"`
	WorkspaceID            string         `json:"workspace_id"`
	EntityType             string         `json:"entity_type"`
	CanonicalName          string         `json:"canonical_name"`
	NormalizedName         string         `json:"normalized_name"`
	Aliases                []string       `json:"aliases"`
	Confidence             float64        `json:"confidence"`
	Provenance             map[string]any `json:"provenance"`
	RepresentativeMemoryID *string        `json:"representative_memory_id,omitempty"`
	Status                 string         `json:"status"`
	MergedInto             *string        `json:"merged_into,omitempty"`
	CreatedAt              time.Time      `json:"created_at"`
	UpdatedAt              time.Time      `json:"updated_at"`
}

// EntityResolution is the result of resolving a surface name to a canonical
// entity.
type EntityResolution struct {
	Entity     Entity  `json:"entity"`
	MatchedVia string  `json:"matched_via"`
	Score      float64 `json:"score"`
}

// MessageContent holds a chat message's body, which the server returns as
// either a plain string or a list of structured content blocks.
type MessageContent struct {
	// Text is set when the message content is a plain string.
	Text string
	// Blocks is set when the message content is a list of content blocks.
	Blocks []ChatMessageContent
}

// IsText reports whether the content is a plain string (no structured blocks).
func (m MessageContent) IsText() bool { return m.Blocks == nil }

// UnmarshalJSON accepts both a JSON string and a JSON array of content blocks.
func (m *MessageContent) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '"' {
		return json.Unmarshal(b, &m.Text)
	}
	return json.Unmarshal(b, &m.Blocks)
}

// MarshalJSON emits a string when only Text is set, otherwise the block list.
func (m MessageContent) MarshalJSON() ([]byte, error) {
	if m.Blocks != nil {
		return json.Marshal(m.Blocks)
	}
	return json.Marshal(m.Text)
}

// ChatMessageContent is a structured content block within a chat message.
type ChatMessageContent struct {
	Type string         `json:"type"`
	Text *string        `json:"text,omitempty"`
	Data map[string]any `json:"data,omitempty"`
}

// ChatMessage is a single message in a chat thread.
type ChatMessage struct {
	ID           string         `json:"id"`
	ThreadID     string         `json:"thread_id"`
	MessageIndex int            `json:"message_index"`
	Role         string         `json:"role"`
	Content      MessageContent `json:"content"`
	Metadata     map[string]any `json:"metadata"`
	CreatedAt    time.Time      `json:"created_at"`
}

// ChatThread is a conversation thread.
type ChatThread struct {
	ID                  string         `json:"id"`
	WorkspaceID         string         `json:"workspace_id"`
	TenantID            string         `json:"tenant_id"`
	UserID              *string        `json:"user_id,omitempty"`
	ContextID           string         `json:"context_id"`
	ObserverID          *string        `json:"observer_id,omitempty"`
	SubjectID           *string        `json:"subject_id,omitempty"`
	Title               *string        `json:"title,omitempty"`
	Metadata            map[string]any `json:"metadata"`
	MessageCount        int            `json:"message_count"`
	LastDecomposedAt    *time.Time     `json:"last_decomposed_at,omitempty"`
	LastDecomposedIndex int            `json:"last_decomposed_index"`
	ExpiresAt           *time.Time     `json:"expires_at,omitempty"`
	CreatedAt           time.Time      `json:"created_at"`
	UpdatedAt           time.Time      `json:"updated_at"`
	Scope               *string        `json:"scope,omitempty"`
	Ownership           string         `json:"ownership"`
	// ParentThread is the id of this thread's parent (chat_threads.parent_thread),
	// or nil for a top-level thread. Set at create time only.
	ParentThread *string `json:"parent_thread,omitempty"`
}

// ChatThreadWithMessages is a thread with its messages inlined.
type ChatThreadWithMessages struct {
	Thread        ChatThread    `json:"thread"`
	Messages      []ChatMessage `json:"messages"`
	TotalMessages int           `json:"total_messages"`
}

// DecompositionResult is the result of triggering thread decomposition.
type DecompositionResult struct {
	ThreadID          string `json:"thread_id"`
	WorkspaceID       string `json:"workspace_id"`
	MessagesProcessed int    `json:"messages_processed"`
	MemoriesCreated   int    `json:"memories_created"`
	FromIndex         int    `json:"from_index"`
	ToIndex           int    `json:"to_index"`
}

// WorkspaceExport is the materialized result of ExportWorkspace.
type WorkspaceExport struct {
	Version           string           `json:"version"`
	WorkspaceID       string           `json:"workspace_id"`
	ExportedAt        string           `json:"exported_at"`
	TotalMemories     int              `json:"total_memories"`
	TotalAssociations int              `json:"total_associations"`
	Memories          []map[string]any `json:"memories"`
	Associations      []map[string]any `json:"associations"`
}
