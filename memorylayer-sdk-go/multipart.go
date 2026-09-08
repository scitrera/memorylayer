package memorylayer

import (
	"bytes"
	"mime/multipart"
	"sort"
)

// encodeMultipart serializes a single-file multipart/form-data body plus string
// fields, returning the encoded bytes and the content-type header (with
// boundary). Both transports ship the raw bytes unchanged, so the multipart
// upload paths work over the Aether transport too.
func encodeMultipart(fileField, filename string, fileData []byte, fields map[string]string) ([]byte, string, error) {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)

	// Write fields in sorted order for deterministic output.
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if err := w.WriteField(k, fields[k]); err != nil {
			return nil, "", err
		}
	}

	fw, err := w.CreateFormFile(fileField, filename)
	if err != nil {
		return nil, "", err
	}
	if _, err := fw.Write(fileData); err != nil {
		return nil, "", err
	}
	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return buf.Bytes(), w.FormDataContentType(), nil
}
