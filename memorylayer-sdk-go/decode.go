package memorylayer

import "encoding/json"

// jsonUnmarshal is a thin alias so files that only need decoding don't import
// encoding/json directly.
func jsonUnmarshal(b []byte, v any) error { return json.Unmarshal(b, v) }

// remapJSON re-encodes an arbitrary decoded value and decodes it into T. It is
// used for the handful of endpoints whose payload may be either nested under a
// wrapper key or returned at the top level (mirroring the Python SDK's
// data.get("key", data) pattern).
func remapJSON[T any](v any) (*T, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out T
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// envelopeOr returns raw["key"] when present (as a sub-document), else raw.
func envelopeOr(raw map[string]any, key string) any {
	if v, ok := raw[key]; ok {
		return v
	}
	return raw
}

// decodeMemoryEnvelope decodes a {"memory": {...}} wrapper or a bare memory.
func decodeMemoryEnvelope(raw map[string]any) (*Memory, error) {
	return remapJSON[Memory](envelopeOr(raw, "memory"))
}

// mustMarshal JSON-encodes v, panicking only on an unencodable value (which the
// SDK never constructs). Used where a marshal error is not actionable.
func mustMarshal(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic("memorylayer: marshal: " + err.Error())
	}
	return b
}

// decodeThreadEnvelope decodes a {"thread": {...}} wrapper or a bare thread.
// The server wraps single-thread responses (create/get/update); decoding one of
// those straight into a ChatThread silently yields a zero-valued struct — an
// empty ID and no error — because the wrapper key is simply unknown to the
// struct. Mirrors decodeMemoryEnvelope.
func decodeThreadEnvelope(raw map[string]any) (*ChatThread, error) {
	return remapJSON[ChatThread](envelopeOr(raw, "thread"))
}

// decodeThreadList decodes a {"threads": [...]} envelope or a bare list.
func decodeThreadList(body []byte) ([]ChatThread, error) {
	return decodeKeyedList[ChatThread](body, "threads")
}

// decodeMessageList decodes a {"messages": [...]} envelope or a bare list.
func decodeMessageList(body []byte) ([]ChatMessage, error) {
	return decodeKeyedList[ChatMessage](body, "messages")
}

// decodeKeyedList decodes either {"<key>": [...]} or a top-level JSON array into
// []T.
func decodeKeyedList[T any](body []byte, key string) ([]T, error) {
	if len(body) == 0 {
		return nil, nil
	}
	trimmed := body
	for len(trimmed) > 0 && (trimmed[0] == ' ' || trimmed[0] == '\n' || trimmed[0] == '\t' || trimmed[0] == '\r') {
		trimmed = trimmed[1:]
	}
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var list []T
		if err := json.Unmarshal(body, &list); err != nil {
			return nil, err
		}
		return list, nil
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, err
	}
	raw, ok := envelope[key]
	if !ok {
		return nil, nil
	}
	var list []T
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, err
	}
	return list, nil
}
