package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

// Document/dataset endpoints require MemoryLayer Enterprise. On OSS servers they
// return an *EnterpriseRequiredError (check with [IsEnterpriseRequired]).

const (
	docMgmtFeature = "Document management"
	docJobsFeature = "Document ingestion jobs"
)

// UploadDocumentOptions configures [Client.UploadDocument].
type UploadDocumentOptions struct {
	TargetContextID  string  // default "_default"
	ChunkingStrategy string  // default "page"
	ChunkSize        int     // default 4096
	ChunkOverlap     int     // default 200
	Importance       float64 // default 0.5
	RetainOriginal   *bool   // default true

	importanceSet bool
}

// WithImportance returns a copy with Importance explicitly set.
func (o UploadDocumentOptions) WithImportance(v float64) UploadDocumentOptions {
	o.Importance = v
	o.importanceSet = true
	return o
}

// UploadDocument uploads a document for ingestion (Enterprise). Returns the
// created document and its ingestion job.
func (c *Client) UploadDocument(ctx context.Context, fileData []byte, filename string, opts UploadDocumentOptions) (*DocumentInfo, *JobInfo, error) {
	targetContextID := orDefault(opts.TargetContextID, "_default")
	chunkingStrategy := orDefault(opts.ChunkingStrategy, "page")
	chunkSize := opts.ChunkSize
	if chunkSize <= 0 {
		chunkSize = 4096
	}
	chunkOverlap := opts.ChunkOverlap
	if chunkOverlap <= 0 {
		chunkOverlap = 200
	}
	importance := opts.Importance
	if !opts.importanceSet && importance == 0 {
		importance = 0.5
	}
	retainOriginal := true
	if opts.RetainOriginal != nil {
		retainOriginal = *opts.RetainOriginal
	}

	fields := map[string]string{
		"target_context_id": targetContextID,
		"chunking_strategy": chunkingStrategy,
		"chunk_size":        strconv.Itoa(chunkSize),
		"chunk_overlap":     strconv.Itoa(chunkOverlap),
		"importance":        strconv.FormatFloat(importance, 'g', -1, 64),
		"retain_original":   boolStr(retainOriginal),
	}
	body, contentType, err := encodeMultipart("file", filename, fileData, fields)
	if err != nil {
		return nil, nil, err
	}

	var out struct {
		Document DocumentInfo `json:"document"`
		Job      JobInfo      `json:"job"`
	}
	if err := c.doUpload(ctx, "/documents", body, contentType, "Document ingestion", &out); err != nil {
		return nil, nil, err
	}
	return &out.Document, &out.Job, nil
}

// ListDocumentsOptions configures [Client.ListDocuments].
type ListDocumentsOptions struct {
	Status string
	// WorkspaceID selects the workspace to list (defaults to the client's
	// configured workspace). The server authz-gates the requested workspace, so a
	// caller can only list workspaces it can read.
	WorkspaceID string
	// DocumentType filters by document type (pdf, markdown, text, html, docx, pptx).
	DocumentType string
	// CreatedAfter/CreatedBefore filter by created_at (ISO datetime).
	CreatedAfter  string
	CreatedBefore string
	Limit         int // default 50 when <= 0
	Offset        int
	// Authority applies a per-call OBO authority (overriding the client default).
	// Mirrors [RecallOptions.Authority]/[GetMemoryOptions.Authority].
	Authority *AuthorityContext
}

// ListDocuments lists documents in the workspace, returning the page and the
// reported total count.
func (c *Client) ListDocuments(ctx context.Context, opts ListDocumentsOptions) ([]DocumentInfo, int, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}}
	if opts.Status != "" {
		q.Set("status", opts.Status)
	}
	if ws := c.resolveWorkspace(opts.WorkspaceID); ws != "" {
		q.Set("workspace_id", ws)
	}
	if opts.DocumentType != "" {
		q.Set("document_type", opts.DocumentType)
	}
	if opts.CreatedAfter != "" {
		q.Set("created_after", opts.CreatedAfter)
	}
	if opts.CreatedBefore != "" {
		q.Set("created_before", opts.CreatedBefore)
	}
	var out struct {
		Documents  []DocumentInfo `json:"documents"`
		TotalCount int            `json:"total_count"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/documents",
		query:             q,
		authority:         opts.Authority,
		enterpriseFeature: docMgmtFeature,
	}, &out); err != nil {
		return nil, 0, err
	}
	total := out.TotalCount
	if total == 0 {
		total = len(out.Documents)
	}
	return out.Documents, total, nil
}

// GetDocument fetches document metadata and processing status.
func (c *Client) GetDocument(ctx context.Context, documentID string) (*DocumentInfo, error) {
	var out DocumentInfo
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/documents/" + documentID,
		enterpriseFeature: docMgmtFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteDocument deletes a document and optionally its extracted memories.
func (c *Client) DeleteDocument(ctx context.Context, documentID string, deleteMemories bool) error {
	return c.doJSON(ctx, requestSpec{
		method:            http.MethodDelete,
		path:              "/documents/" + documentID,
		query:             url.Values{"delete_memories": {boolStr(deleteMemories)}},
		enterpriseFeature: docMgmtFeature,
	}, nil)
}

// SearchDocumentPages searches document pages via ColPali MaxSim visual
// similarity (Enterprise).
func (c *Client) SearchDocumentPages(ctx context.Context, query string, limit int, docIDs []string) (*PageSearchResult, error) {
	if limit <= 0 {
		limit = 10
	}
	payload := map[string]any{"query": query, "limit": limit}
	if len(docIDs) > 0 {
		payload["doc_ids"] = docIDs
	}
	var out PageSearchResult
	if err := c.requestJSON(ctx, http.MethodPost, "/documents/search", payload, requestSpec{
		enterpriseFeature: "Document page search",
	}, &out); err != nil {
		return nil, err
	}
	if out.Query == "" {
		out.Query = query
	}
	return &out, nil
}

// GetDocumentPages returns all pages for a document.
func (c *Client) GetDocumentPages(ctx context.Context, documentID string) ([]DocumentPage, error) {
	var out struct {
		Pages []DocumentPage `json:"pages"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              fmt.Sprintf("/documents/%s/pages", documentID),
		enterpriseFeature: "Document pages",
	}, &out); err != nil {
		return nil, err
	}
	return out.Pages, nil
}

// GetPageImage returns a page image as raw PNG bytes (Enterprise).
func (c *Client) GetPageImage(ctx context.Context, documentID, pageID string) ([]byte, error) {
	return c.doDownload(ctx, fmt.Sprintf("/documents/%s/pages/%s/image", documentID, pageID), "Document page images")
}

// GetJob returns ingestion job status and progress.
func (c *Client) GetJob(ctx context.Context, jobID string) (*JobInfo, error) {
	var out JobInfo
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/documents/jobs/" + jobID,
		enterpriseFeature: docJobsFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListJobs lists ingestion jobs in the workspace.
func (c *Client) ListJobs(ctx context.Context, status string, limit int) ([]JobInfo, error) {
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}}
	if status != "" {
		q.Set("status", status)
	}
	var out struct {
		Jobs []JobInfo `json:"jobs"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/documents/jobs",
		query:             q,
		enterpriseFeature: docJobsFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Jobs, nil
}

// CancelJob cancels a queued or running ingestion job.
func (c *Client) CancelJob(ctx context.Context, jobID string) error {
	return c.doJSON(ctx, requestSpec{
		method:            http.MethodPost,
		path:              fmt.Sprintf("/documents/jobs/%s/cancel", jobID),
		enterpriseFeature: docJobsFeature,
	}, nil)
}

// ReprocessDocumentOptions configures [Client.ReprocessDocument]. Nil/zero
// fields are omitted (server keeps the original extraction options).
type ReprocessDocumentOptions struct {
	TargetContextID  string
	ChunkingStrategy string
	ChunkSize        *int
	ChunkOverlap     *int
	Importance       *float64
}

// ReprocessDocument reprocesses a document with optionally different extraction
// options.
func (c *Client) ReprocessDocument(ctx context.Context, documentID string, opts ReprocessDocumentOptions) (*JobInfo, error) {
	payload := map[string]any{}
	if opts.TargetContextID != "" {
		payload["target_context_id"] = opts.TargetContextID
	}
	if opts.ChunkingStrategy != "" {
		payload["chunking_strategy"] = opts.ChunkingStrategy
	}
	if opts.ChunkSize != nil {
		payload["chunk_size"] = *opts.ChunkSize
	}
	if opts.ChunkOverlap != nil {
		payload["chunk_overlap"] = *opts.ChunkOverlap
	}
	if opts.Importance != nil {
		payload["importance"] = *opts.Importance
	}
	var body any
	if len(payload) > 0 {
		body = payload
	}
	var out JobInfo
	if err := c.requestJSON(ctx, http.MethodPost, fmt.Sprintf("/documents/%s/reprocess", documentID), body, requestSpec{
		enterpriseFeature: "Document reprocessing",
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// doUpload posts a multipart body, mapping 501 to an EnterpriseRequiredError for
// feature, then decodes into out.
func (c *Client) doUpload(ctx context.Context, path string, body []byte, contentType, feature string, out any) error {
	resp, err := c.doRaw(ctx, requestSpec{
		method:      http.MethodPost,
		path:        path,
		body:        body,
		contentType: contentType,
	})
	if err != nil {
		return err
	}
	if err := mapError(resp, feature); err != nil {
		return err
	}
	if out != nil && len(resp.Body) > 0 {
		return jsonUnmarshal(resp.Body, out)
	}
	return nil
}

// doDownload GETs a binary body, mapping 501 to an EnterpriseRequiredError for
// feature.
func (c *Client) doDownload(ctx context.Context, path, feature string) ([]byte, error) {
	resp, err := c.doRaw(ctx, requestSpec{method: http.MethodGet, path: path})
	if err != nil {
		return nil, err
	}
	if err := mapError(resp, feature); err != nil {
		return nil, err
	}
	return resp.Body, nil
}

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}
