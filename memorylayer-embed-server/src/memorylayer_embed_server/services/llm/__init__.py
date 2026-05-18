"""LLM inference profiles for the embed-server.

When ``MEMORYLAYER_EMBED_LLM_ENABLED=true``, the embed-server hosts one
or more ``vllm serve`` child processes (each a chat LLM) and exposes
OpenAI-compatible chat / completions / models endpoints. Multiple
profiles can run concurrently on auto-assigned loopback ports; routing
is by the OpenAI-standard ``model`` request field.
"""
