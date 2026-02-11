# /memorylayer-remember

Quick command to store a memory.

## Usage

```
/memorylayer-remember <content>
```

## Examples

```
/memorylayer-remember We decided to use PostgreSQL for better JSON support
/memorylayer-remember The auth bug was caused by missing token refresh
/memorylayer-remember User prefers detailed commit messages
```

## Behavior

When this command is invoked:

1. Take the provided content and store it using `memory_remember`
2. Auto-detect appropriate settings:
   - **Type**: Infer from content (decision → semantic, fix → procedural, preference → semantic)
   - **Importance**: Default to 0.7, adjust based on keywords (critical → 0.9, minor → 0.5)
   - **Tags**: Extract from content (technology names, component names)

3. Confirm storage with memory ID

## Response

```
Stored memory: mem_abc123
Type: semantic | Importance: 0.7
Tags: postgresql, database, architecture
```
