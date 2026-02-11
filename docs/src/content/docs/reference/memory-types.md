---
title: Memory Types & Subtypes Reference
description: Complete reference for all memory types and subtypes
sidebar:
  order: 2
  label: Memory Types & Subtypes
---

## Cognitive Types

| Type | Enum Value | Description | Retention | Typical Use |
|------|-----------|-------------|-----------|-------------|
| Episodic | `episodic` | Specific events/interactions | Decays over time | Event logs, interaction history |
| Semantic | `semantic` | Facts, concepts, relationships | Permanent until modified | Preferences, decisions, knowledge |
| Procedural | `procedural` | How-to knowledge | Permanent | Solutions, patterns, workflows |
| Working | `working` | Current task context | Session-scoped (TTL) | Active task tracking |

### Python

```python
from memorylayer import MemoryType

MemoryType.EPISODIC    # "episodic"
MemoryType.SEMANTIC    # "semantic"
MemoryType.PROCEDURAL  # "procedural"
MemoryType.WORKING     # "working"
```

### TypeScript

```typescript
import { MemoryType } from "@scitrera/memorylayer-sdk";

MemoryType.EPISODIC    // "episodic"
MemoryType.SEMANTIC    // "semantic"
MemoryType.PROCEDURAL  // "procedural"
MemoryType.WORKING     // "working"
```

## Domain Subtypes

| Subtype | Enum Value | Description |
|---------|-----------|-------------|
| Solution | `solution` | Working fixes to problems |
| Problem | `problem` | Issues encountered |
| CodePattern | `code_pattern` | Reusable code patterns |
| Fix | `fix` | Bug fixes with context |
| Error | `error` | Error patterns and resolutions |
| Workflow | `workflow` | Process knowledge |
| Preference | `preference` | User or project preferences |
| Decision | `decision` | Architectural decisions |
| Profile | `profile` | Person or entity profiles |
| Entity | `entity` | Named entities (people, places, things) |
| Event | `event` | Significant events or milestones |
| Directive | `directive` | User instructions and constraints |

### Python

```python
from memorylayer import MemorySubtype

MemorySubtype.SOLUTION       # "solution"
MemorySubtype.PROBLEM        # "problem"
MemorySubtype.CODE_PATTERN   # "code_pattern"
MemorySubtype.FIX            # "fix"
MemorySubtype.ERROR          # "error"
MemorySubtype.WORKFLOW       # "workflow"
MemorySubtype.PREFERENCE     # "preference"
MemorySubtype.DECISION       # "decision"
MemorySubtype.PROFILE        # "profile"
MemorySubtype.ENTITY         # "entity"
MemorySubtype.EVENT          # "event"
MemorySubtype.DIRECTIVE      # "directive"
```

### TypeScript

```typescript
import { MemorySubtype } from "@scitrera/memorylayer-sdk";

MemorySubtype.SOLUTION       // "solution"
MemorySubtype.PROBLEM        // "problem"
MemorySubtype.CODE_PATTERN   // "code_pattern"
MemorySubtype.FIX            // "fix"
MemorySubtype.ERROR          // "error"
MemorySubtype.WORKFLOW       // "workflow"
MemorySubtype.PREFERENCE     // "preference"
MemorySubtype.DECISION       // "decision"
MemorySubtype.DIRECTIVE      // "directive"
MemorySubtype.PROFILE        // "profile"
MemorySubtype.ENTITY         // "entity"
MemorySubtype.EVENT          // "event"
```

## Recall Modes

| Mode | Enum Value | Description |
|------|-----------|-------------|
| RAG | `rag` | Vector similarity search |
| LLM | `llm` | LLM-powered semantic search |
| Hybrid | `hybrid` | Combination of RAG and LLM |

## Importance Scoring Guide

| Score | Level | Use For |
|-------|-------|---------|
| 0.9–1.0 | Critical | Security fixes, breaking changes, core architecture |
| 0.7–0.8 | High | Bug fixes, API changes, important patterns |
| 0.5–0.6 | Standard | Preferences, general knowledge |
| 0.3–0.4 | Low | Observations, temporary workarounds |
| 0.0–0.2 | Ephemeral | Debugging traces, session context |
