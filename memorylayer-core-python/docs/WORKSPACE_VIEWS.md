# Workspace views

MemoryLayer workspaces are durable logical project or knowledge scopes. A
workspace view is one checkout, Git worktree, snapshot, overlay, or directory
through which an agent sees that logical workspace.

This distinction lets several agents share workspace-level memory while keeping
their working-tree facts separate. There is no workspace-wide "latest checkout"
record: latest observed state is keyed by `(workspace_id, view_id, observer_id)`.

## Authority boundary

- MemoryLayer stores durable workspace/view identity, current observation heads,
  immutable revisions, and cursor-paginated history.
- A surface such as Sahara observes local Git/directory state and publishes it.
- Aether owns live tool-host presence, leases, task routing, and scheduling.
- The tool host owns actual filesystem permission. MemoryLayer never treats a
  client-supplied absolute path as authority.

View observations accept an opaque `root_ref`, which only the associated tool
host may interpret. The schema intentionally has no absolute-path field.

## View API

Views live beneath their parent workspace:

```text
POST /v1/workspaces/{workspace_id}/views
GET  /v1/workspaces/{workspace_id}/views
GET  /v1/workspaces/{workspace_id}/views/{view_id}
PUT  /v1/workspaces/{workspace_id}/views/{view_id}
DELETE /v1/workspaces/{workspace_id}/views/{view_id}
POST /v1/workspaces/{workspace_id}/views/{view_id}/restore
GET  /v1/workspaces/{workspace_id}/views/{view_id}/revisions
```

Create requires `If-None-Match: *` and `Idempotency-Key`. Replace requires the
current `If-Match` ETag and `Idempotency-Key`. Successful mutations and reads
return the current `ETag`. Lists and revision history use opaque, scope-bound
`page_token` cursors. Delete writes a durable tombstone; restore reactivates the
same stable view and retains its immutable history.

Example create body:

```json
{
  "view_id": "worktree-feature-a",
  "kind": "git_worktree",
  "display_name": "Feature A worktree",
  "memory_context_id": "ctx-worktree-feature-a",
  "capabilities": ["workspace.read", "workspace.write"]
}
```

`memory_context_id` is optional. It can scope memories that describe only this
view or revision; workspace-wide project knowledge remains in the containing
workspace.

## Observation API

Each observer publishes its current view state through a stable `observer_id`:

```text
PUT /v1/workspaces/{workspace_id}/views/{view_id}/observations/{observer_id}
GET /v1/workspaces/{workspace_id}/views/{view_id}/observations
GET /v1/workspaces/{workspace_id}/views/{view_id}/observations/{observer_id}
DELETE /v1/workspaces/{workspace_id}/views/{view_id}/observations/{observer_id}
POST /v1/workspaces/{workspace_id}/views/{view_id}/observations/{observer_id}/restore
GET /v1/workspaces/{workspace_id}/views/{view_id}/observations/{observer_id}/revisions
```

The first PUT uses `If-None-Match: *`; later PUTs use the current `If-Match`
ETag. Both require an idempotency key.

```json
{
  "generation": "sahara-process-7b55",
  "sequence": 12,
  "tool_host_id": "tui-host-01",
  "execution_site": "client",
  "root_ref": "local-root-1",
  "capabilities": ["workspace.read", "workspace.write"],
  "vcs": {
    "kind": "git",
    "repository_id": "git:opaque-fingerprint",
    "worktree_id": "git:opaque-worktree-id",
    "head_revision": "0123456789abcdef",
    "branch": "feature/a",
    "dirty": true
  }
}
```

`sequence` must advance within one observer `generation`. A restarted observer
uses a new generation and may restart its sequence. This detects stale writes in
addition to ETag compare-and-swap. Capability lists are normalized to a sorted,
unique wire form.

An observation may record the last associated `tool_host_id`, but it does not
assert that the host is still online. Callers must resolve current reachability
through the live router before invoking tools. Background or scheduled work must
bind to an eligible worker host instead of assuming an interactive client is
connected.
