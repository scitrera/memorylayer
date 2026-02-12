---
title: CLI Reference
description: MemoryLayer command-line interface
sidebar:
  order: 4
---

The `memorylayer` CLI is provided by the `memorylayer-server` package.

## Installation

```bash
pip install memorylayer-server
```

## Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--verbose` / `-v` | Enable debug logging | `false` |

## Commands

### memorylayer serve

Start the HTTP REST API server.

```bash
memorylayer serve [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Port to listen on | `61001` |
| `--host` | Host to bind to | `127.0.0.1` |

**Examples:**

```bash
# Start on default port
memorylayer serve

# Custom port
memorylayer serve --port 8080

# Debug mode
memorylayer serve --verbose
```

### memorylayer version

Show version information.

```bash
memorylayer version
```

[//]: # (### memorylayer info)

[//]: # ()
[//]: # (Show system information and configuration.)

[//]: # ()
[//]: # (```bash)

[//]: # (memorylayer info [OPTIONS])

[//]: # (```)

[//]: # ()
[//]: # (**Options:**)

[//]: # ()
[//]: # (| Option | Description | Default |)

[//]: # (|--------|-------------|---------|)

[//]: # (| `--format` | Output format &#40;`text` or `json`&#41; | `text` |)

### memorylayer --help

Show help information.

```bash
memorylayer --help
```

## Usage with Claude Code

The CLI is used to run the HTTP server:

**HTTP Server** — Run `memorylayer serve` in a terminal, then connect via SDK or the separate MCP server package (`@scitrera/memorylayer-mcp-server`).

The MCP server is a separate TypeScript package, not part of the core server CLI.

See [Claude Code Integration](/integrations/claude-code/) for detailed setup instructions.
