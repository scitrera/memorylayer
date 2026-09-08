# memorylayer-server-rpg

Repository Planning Graph (RPG) services for `memorylayer-server`.

The package adds repository-structure graph sync, search, traversal, overlays,
conflict detection, maintenance, and optional LLM enrichment under `/v1/rpg`.
RPG nodes are stored as MemoryLayer memories and RPG relationships as normal
associations, so the feature works with every compatible storage backend.

## Install

Requires Python 3.12 or newer. Installing the plugin also installs the matching
`memorylayer-server` version.

```bash
pip install memorylayer-server-rpg
memorylayer serve
```

Installation is sufficient: the server discovers the package through the
`memorylayer_server.plugin_packages` entry point and registers its API routes,
task handlers, service, and ontology contribution automatically.

The core server wheel does not bundle this plugin. The official server Docker
image already includes it.

The official Python and TypeScript SDKs expose this surface as `client.rpg`
(`sync_client.rpg` for synchronous Python); the Go SDK exposes `client.Rpg`.

## License

Apache 2.0. See [LICENSE](./LICENSE).
