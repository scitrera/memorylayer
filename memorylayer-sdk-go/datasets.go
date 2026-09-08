package memorylayer

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
)

const (
	datasetMgmtFeature = "Dataset management"
	datasetJobsFeature = "Dataset processing jobs"
)

// UploadDatasetOptions configures [Client.UploadDataset].
type UploadDatasetOptions struct {
	Name              string  // defaults to filename stem server-side
	TargetContextID   string  // default "_default"
	Importance        float64 // default 0.5
	SampleRows        int     // default 1000
	DetectTimeSeries  *bool   // default true
	GenerateSummaries *bool   // default true

	importanceSet bool
}

// WithImportance returns a copy with Importance explicitly set.
func (o UploadDatasetOptions) WithImportance(v float64) UploadDatasetOptions {
	o.Importance = v
	o.importanceSet = true
	return o
}

// UploadDataset uploads a dataset for profiling and memory extraction
// (Enterprise). Returns the created dataset and its processing job.
func (c *Client) UploadDataset(ctx context.Context, fileData []byte, filename string, opts UploadDatasetOptions) (*DatasetInfo, *DatasetJobInfo, error) {
	targetContextID := orDefault(opts.TargetContextID, "_default")
	importance := opts.Importance
	if !opts.importanceSet && importance == 0 {
		importance = 0.5
	}
	sampleRows := opts.SampleRows
	if sampleRows <= 0 {
		sampleRows = 1000
	}
	detectTimeSeries := true
	if opts.DetectTimeSeries != nil {
		detectTimeSeries = *opts.DetectTimeSeries
	}
	generateSummaries := true
	if opts.GenerateSummaries != nil {
		generateSummaries = *opts.GenerateSummaries
	}

	fields := map[string]string{
		"target_context_id":  targetContextID,
		"importance":         strconv.FormatFloat(importance, 'g', -1, 64),
		"sample_rows":        strconv.Itoa(sampleRows),
		"detect_time_series": boolStr(detectTimeSeries),
		"generate_summaries": boolStr(generateSummaries),
	}
	if opts.Name != "" {
		fields["name"] = opts.Name
	}
	body, contentType, err := encodeMultipart("file", filename, fileData, fields)
	if err != nil {
		return nil, nil, err
	}

	var out struct {
		Dataset DatasetInfo    `json:"dataset"`
		Job     DatasetJobInfo `json:"job"`
	}
	if err := c.doUpload(ctx, "/datasets", body, contentType, datasetMgmtFeature, &out); err != nil {
		return nil, nil, err
	}
	return &out.Dataset, &out.Job, nil
}

// ListDatasetsOptions configures [Client.ListDatasets].
type ListDatasetsOptions struct {
	Status string
	Limit  int // default 50 when <= 0
	Offset int
}

// ListDatasets lists datasets in the workspace, returning the page and the
// reported total count.
func (c *Client) ListDatasets(ctx context.Context, opts ListDatasetsOptions) ([]DatasetInfo, int, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}, "offset": {strconv.Itoa(opts.Offset)}}
	if opts.Status != "" {
		q.Set("status", opts.Status)
	}
	var out struct {
		Datasets   []DatasetInfo `json:"datasets"`
		TotalCount int           `json:"total_count"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/datasets",
		query:             q,
		enterpriseFeature: datasetMgmtFeature,
	}, &out); err != nil {
		return nil, 0, err
	}
	total := out.TotalCount
	if total == 0 {
		total = len(out.Datasets)
	}
	return out.Datasets, total, nil
}

// GetDataset fetches dataset metadata, schema, and profile.
func (c *Client) GetDataset(ctx context.Context, datasetID string) (*DatasetInfo, error) {
	var out DatasetInfo
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/datasets/" + datasetID,
		enterpriseFeature: datasetMgmtFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteDataset deletes a dataset and optionally its extracted memories.
func (c *Client) DeleteDataset(ctx context.Context, datasetID string, deleteMemories bool) error {
	return c.doJSON(ctx, requestSpec{
		method:            http.MethodDelete,
		path:              "/datasets/" + datasetID,
		query:             url.Values{"delete_memories": {boolStr(deleteMemories)}},
		enterpriseFeature: datasetMgmtFeature,
	}, nil)
}

// GetDatasetMemories returns memories extracted from a dataset (raw maps).
func (c *Client) GetDatasetMemories(ctx context.Context, datasetID string) ([]map[string]any, error) {
	var out struct {
		Memories []map[string]any `json:"memories"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              fmt.Sprintf("/datasets/%s/memories", datasetID),
		enterpriseFeature: datasetMgmtFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Memories, nil
}

// QueryDatasetSliceOptions configures [Client.QueryDatasetSlice].
type QueryDatasetSliceOptions struct {
	SQL        string
	Columns    []string
	Filters    []map[string]any
	OrderBy    string
	Descending bool
	Limit      int // default 100 when <= 0
	Offset     int
}

// QueryDatasetSlice queries a slice of dataset data using DuckDB (Enterprise).
func (c *Client) QueryDatasetSlice(ctx context.Context, datasetID string, opts QueryDatasetSliceOptions) (*DatasetSliceResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 100
	}
	payload := map[string]any{
		"limit":      limit,
		"offset":     opts.Offset,
		"descending": opts.Descending,
	}
	if opts.SQL != "" {
		payload["sql"] = opts.SQL
	}
	if opts.Columns != nil {
		payload["columns"] = opts.Columns
	}
	if opts.Filters != nil {
		payload["filters"] = opts.Filters
	}
	if opts.OrderBy != "" {
		payload["order_by"] = opts.OrderBy
	}
	var out DatasetSliceResult
	if err := c.requestJSON(ctx, http.MethodPost, fmt.Sprintf("/datasets/%s/slice", datasetID), payload, requestSpec{
		enterpriseFeature: datasetMgmtFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// GetDatasetJob returns dataset processing job status and progress.
func (c *Client) GetDatasetJob(ctx context.Context, jobID string) (*DatasetJobInfo, error) {
	var out DatasetJobInfo
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/datasets/jobs/" + jobID,
		enterpriseFeature: datasetJobsFeature,
	}, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ListDatasetJobs lists dataset processing jobs in the workspace.
func (c *Client) ListDatasetJobs(ctx context.Context, status string, limit int) ([]DatasetJobInfo, error) {
	if limit <= 0 {
		limit = 50
	}
	q := url.Values{"limit": {strconv.Itoa(limit)}}
	if status != "" {
		q.Set("status", status)
	}
	var out struct {
		Jobs []DatasetJobInfo `json:"jobs"`
	}
	if err := c.doJSON(ctx, requestSpec{
		method:            http.MethodGet,
		path:              "/datasets/jobs",
		query:             q,
		enterpriseFeature: datasetJobsFeature,
	}, &out); err != nil {
		return nil, err
	}
	return out.Jobs, nil
}

// CancelDatasetJob cancels a queued or running dataset processing job.
func (c *Client) CancelDatasetJob(ctx context.Context, jobID string) error {
	return c.doJSON(ctx, requestSpec{
		method:            http.MethodPost,
		path:              fmt.Sprintf("/datasets/jobs/%s/cancel", jobID),
		enterpriseFeature: datasetJobsFeature,
	}, nil)
}
