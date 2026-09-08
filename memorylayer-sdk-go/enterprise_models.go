package memorylayer

import "time"

// DocumentPage is a page from an ingested document (Enterprise).
type DocumentPage struct {
	ID               string         `json:"id"`
	DocumentID       string         `json:"document_id"`
	WorkspaceID      string         `json:"workspace_id"`
	PageNo           int            `json:"page_no"`
	ImageStoragePath *string        `json:"image_storage_path,omitempty"`
	Transcript       *string        `json:"transcript,omitempty"`
	TranscriptModel  *string        `json:"transcript_model,omitempty"`
	Metadata         map[string]any `json:"metadata"`
	CreatedAt        *time.Time     `json:"created_at,omitempty"`
	RelevanceScore   *float64       `json:"relevance_score,omitempty"`
}

// DocumentInfo is document metadata returned from the API (Enterprise).
type DocumentInfo struct {
	ID                    string         `json:"id"`
	WorkspaceID           string         `json:"workspace_id"`
	Filename              string         `json:"filename"`
	DocumentType          string         `json:"document_type"`
	ContentHash           string         `json:"content_hash"`
	SizeBytes             int64          `json:"size_bytes"`
	MimeType              *string        `json:"mime_type,omitempty"`
	Status                string         `json:"status"`
	SourceVfsRef          *string        `json:"source_vfs_ref,omitempty"`
	TargetContextID       string         `json:"target_context_id"`
	PageCount             int            `json:"page_count"`
	ChunkCount            int            `json:"chunk_count"`
	MemoryIDs             []string       `json:"memory_ids"`
	StoragePath           *string        `json:"storage_path,omitempty"`
	RetainOriginal        bool           `json:"retain_original"`
	Metadata              map[string]any `json:"metadata"`
	CreatedAt             time.Time      `json:"created_at"`
	ProcessingStartedAt   *time.Time     `json:"processing_started_at,omitempty"`
	ProcessingCompletedAt *time.Time     `json:"processing_completed_at,omitempty"`
}

// JobInfo is an ingestion job status (Enterprise).
type JobInfo struct {
	ID                   string           `json:"id"`
	WorkspaceID          string           `json:"workspace_id"`
	DocumentIDs          []string         `json:"document_ids"`
	Status               string           `json:"status"`
	ProgressPercent      int              `json:"progress_percent"`
	DocumentsProcessed   int              `json:"documents_processed"`
	TotalMemoriesCreated int              `json:"total_memories_created"`
	Errors               []map[string]any `json:"errors"`
	CreatedAt            time.Time        `json:"created_at"`
	StartedAt            *time.Time       `json:"started_at,omitempty"`
	CompletedAt          *time.Time       `json:"completed_at,omitempty"`
}

// PageSearchResult is the result of a document page search (Enterprise).
type PageSearchResult struct {
	Pages      []DocumentPage `json:"pages"`
	TotalCount int            `json:"total_count"`
	Query      string         `json:"query"`
}

// DatasetColumn is column-level schema and statistics from dataset profiling
// (Enterprise).
type DatasetColumn struct {
	Name        string  `json:"name"`
	Dtype       string  `json:"dtype"`
	ColumnType  string  `json:"column_type"`
	Nullable    bool    `json:"nullable"`
	NullCount   int     `json:"null_count"`
	NullPercent float64 `json:"null_percent"`
	UniqueCount int     `json:"unique_count"`

	MinValue    *float64 `json:"min_value,omitempty"`
	MaxValue    *float64 `json:"max_value,omitempty"`
	MeanValue   *float64 `json:"mean_value,omitempty"`
	MedianValue *float64 `json:"median_value,omitempty"`
	StdValue    *float64 `json:"std_value,omitempty"`
	P25Value    *float64 `json:"p25_value,omitempty"`
	P75Value    *float64 `json:"p75_value,omitempty"`

	MinLength *int     `json:"min_length,omitempty"`
	MaxLength *int     `json:"max_length,omitempty"`
	AvgLength *float64 `json:"avg_length,omitempty"`

	TopValues []map[string]any `json:"top_values,omitempty"`

	IsTemporal         bool    `json:"is_temporal"`
	TemporalResolution *string `json:"temporal_resolution,omitempty"`
	TemporalRangeStart *string `json:"temporal_range_start,omitempty"`
	TemporalRangeEnd   *string `json:"temporal_range_end,omitempty"`

	Histogram map[string]any `json:"histogram,omitempty"`
}

// DatasetInfo is dataset metadata returned from the API (Enterprise).
type DatasetInfo struct {
	ID                   string          `json:"id"`
	WorkspaceID          string          `json:"workspace_id"`
	Name                 string          `json:"name"`
	Filename             string          `json:"filename"`
	Format               string          `json:"format"`
	ContentHash          string          `json:"content_hash"`
	SizeBytes            int64           `json:"size_bytes"`
	Status               string          `json:"status"`
	TargetContextID      string          `json:"target_context_id"`
	RowCount             int             `json:"row_count"`
	ColumnCount          int             `json:"column_count"`
	Columns              []DatasetColumn `json:"columns"`
	MemoryIDs            []string        `json:"memory_ids"`
	ProfileSummary       *string         `json:"profile_summary,omitempty"`
	Metadata             map[string]any  `json:"metadata"`
	CreatedAt            time.Time       `json:"created_at"`
	ProfilingStartedAt   *time.Time      `json:"profiling_started_at,omitempty"`
	ProfilingCompletedAt *time.Time      `json:"profiling_completed_at,omitempty"`
}

// DatasetJobInfo is a dataset processing job status (Enterprise).
type DatasetJobInfo struct {
	ID                   string           `json:"id"`
	WorkspaceID          string           `json:"workspace_id"`
	DatasetIDs           []string         `json:"dataset_ids"`
	Status               string           `json:"status"`
	ProgressPercent      int              `json:"progress_percent"`
	DatasetsProcessed    int              `json:"datasets_processed"`
	TotalMemoriesCreated int              `json:"total_memories_created"`
	Errors               []map[string]any `json:"errors"`
	CreatedAt            time.Time        `json:"created_at"`
	StartedAt            *time.Time       `json:"started_at,omitempty"`
	CompletedAt          *time.Time       `json:"completed_at,omitempty"`
}

// DatasetSliceResult is the result of a dataset slice query (Enterprise).
type DatasetSliceResult struct {
	DatasetID     string   `json:"dataset_id"`
	Columns       []string `json:"columns"`
	Dtypes        []string `json:"dtypes"`
	Rows          [][]any  `json:"rows"`
	TotalMatching int      `json:"total_matching"`
	ReturnedCount int      `json:"returned_count"`
	SQLExecuted   *string  `json:"sql_executed,omitempty"`
}
