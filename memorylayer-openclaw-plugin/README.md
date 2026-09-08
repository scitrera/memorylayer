# `@scitrera/memorylayer-openclaw-plugin`

OpenClaw plugin that backs OpenClaw's first-class memory capability with
[MemoryLayer.ai](https://memorylayer.ai). Sibling to
[`memorylayer-cc-plugin`](../memorylayer-cc-plugin) (Claude Code) and
[`memorylayer-opencode-plugin`](../memorylayer-opencode-plugin) (OpenCode).

## Architecture

The plugin runs inside an OpenClaw container that's launched as part of a
per-principal sandbox. All MemoryLayer traffic is tunneled through the
per-sandbox Aether sidecar relay using
[`AetherFetchTransport`](https://github.com/scitrera/aether) — the plugin
holds **no real credentials**. The sidecar terminates the relay locally and
injects the proper Aether identity upstream.

```
OpenClaw  ──┬── plugin: @scitrera/memorylayer-openclaw-plugin
            │       └── @scitrera/memorylayer-sdk
            │             └── AetherFetchTransport (custom `fetch`)
            │                   └── @scitrera/aether-client
            │                         └── ProxyHttp → localhost:<sidecar-port>
            │
            └── sidecar (Aether relay + identity injection)
                  └── upstream Aether → MemoryLayer service
```

## Required env vars

The sandbox-provider sets these when launching the OpenClaw container:

| Variable                       | Required | Description                                                |
| ------------------------------ | -------- | ---------------------------------------------------------- |
| `AETHER_GATEWAY_ADDRESS`       | yes      | host:port of the sidecar Aether relay                      |
| `AETHER_WORKSPACE`             | yes      | Aether workspace identity                                  |
| `AETHER_IMPLEMENTATION`        | yes      | Aether implementation identity                             |
| `AETHER_SPECIFIER`             | no       | Aether specifier (default: `default`)                      |
| `MEMORYLAYER_TARGET_TOPIC`     | yes      | Aether target topic of MemoryLayer service                 |
| `MEMORYLAYER_BASE_URL`         | no       | Nominal base URL passed to the SDK                          |
| `MEMORYLAYER_TIMEOUT_MS`       | no       | Per-request timeout (default: 30000)                       |
| `WORKCLAW_TENANT_ID`           | yes      | Tenant the principal belongs to                            |
| `WORKCLAW_PRINCIPAL_TYPE`      | yes      | `user` or `agent`                                          |
| `WORKCLAW_PRINCIPAL_ID`        | yes      | Principal identifier                                       |
| `WORKCLAW_DEFAULT_WORKSPACE`   | no       | Default MemoryLayer workspace id                           |
| `WORKCLAW_GRANT_ID`            | no       | Default Aether grant id (sets default OBO authority)       |
| `MEMORYLAYER_SKILLS_SYNC`      | no       | `1`/`true` to materialize MemoryLayer skills into OpenClaw (default OFF) |

## Skills sync (opt-in)

When `MEMORYLAYER_SKILLS_SYNC=1`, on each agent bootstrap the plugin fetches the
principal's **enabled** MemoryLayer skills and calls the SDK's
`client.skills.materialize(...)` to write each `SKILL.md` + its files into the
agent workspace skills directory (`<workspaceDir>/skills`). OpenClaw already
scans **and** watches that directory, so the materialized skills are discovered
and hot-reloaded natively — MemoryLayer is the source of truth, OpenClaw the
local cache. The sync is best-effort: errors are logged and never block agent
bootstrap.

Because `<workspaceDir>/skills` is a dir OpenClaw watches by default, **no
`openclaw.json` change is required**. If a deployment ever materializes into a
non-default directory instead, that directory must be declared so OpenClaw scans
it, via one line in the rendered `openclaw.json`:

```jsonc
{ "skills": { "load": { "extraDirs": ["/abs/path/to/dir"] } } }
```

(The sandbox-provider renders `openclaw.json` in
`internal/openclaw/bootstrap.go`; this plugin does not edit it.)

## OpenClaw integration

Loaded via OpenClaw's plugin manifest mechanism — package.json's
`openclaw.extensions` field points at `./dist/bundle/index.js`, which exports
the `definePluginEntry(...)` result. The plugin registers a single
`MemoryPluginCapability` covering `runtime`, `flushPlanResolver`,
`promptBuilder`, and `publicArtifacts`.

## Supported host and current behavior

The current build targets OpenClaw 2026.9.3 or newer and Node.js 24.16 or newer.
Memory search and reads use MemoryLayer recall and memory retrieval through the
Aether transport. Message persistence and opt-in skills synchronization are
implemented and tested. The flush-plan resolver, prompt builder, and public
artifact provider currently return no additional content.

## Develop

```bash
npm install
npm run build       # tsc → dist/
npm test            # vitest
```

## License

Apache-2.0
