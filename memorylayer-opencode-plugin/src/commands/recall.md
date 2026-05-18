# /memorylayer-recall

Quick command to search memories.

## Usage

```
/memorylayer-recall <query>
```

## Examples

```
/memorylayer-recall database decisions
/memorylayer-recall authentication bugs
/memorylayer-recall user preferences
/memorylayer-recall what do we know about the API
```

## Behavior

When this command is invoked:

1. Use `memory_recall` with the provided query
2. Display results in a readable format
3. Show relevance scores and key metadata

## Response Format

```
Found 3 relevant memories:

1. [0.89] We decided to use PostgreSQL for better JSON support
   Type: semantic | Tags: postgresql, database
   Stored: 2 days ago

2. [0.76] Database connection pooling set to max 20 connections
   Type: procedural | Tags: database, config
   Stored: 1 week ago

3. [0.71] User prefers PostgreSQL over MySQL for new projects
   Type: semantic | Tags: database, preference
   Stored: 3 weeks ago
```

If no results:

```
No memories found matching "your query"

Try:
- Broader search terms
- Check /memorylayer-status for connection issues
- Use /memorylayer-remember to store new memories
```
