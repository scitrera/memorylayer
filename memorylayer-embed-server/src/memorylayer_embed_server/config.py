"""Configuration constants for MemoryLayer Embed Server."""

# ============================================
# Server Configuration
# ============================================
EMBED_SERVER_HOST = 'MEMORYLAYER_EMBED_SERVER_HOST'
DEFAULT_EMBED_SERVER_HOST = '0.0.0.0'
EMBED_SERVER_PORT = 'MEMORYLAYER_EMBED_SERVER_PORT'
DEFAULT_EMBED_SERVER_PORT = 61051

# ============================================
# Transcription
# ============================================
EMBED_SERVER_TRANSCRIPTION_ENABLED = 'MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED'
DEFAULT_EMBED_SERVER_TRANSCRIPTION_ENABLED = True

EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT = 'MEMORYLAYER_EMBED_TRANSCRIPTION_SYSTEM_PROMPT'
DEFAULT_EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT = (
    "You are a document transcription assistant. Your task is to convert the provided document "
    "into clean, well-formatted Markdown. Accurately preserve the original document's structure, "
    "including all headings, lists, bold and italic text, links, tables, and code blocks. "
    "Provide appropriate placeholder references for plots and figures with some description "
    "to enable interpretation in the absence of the plots/figures. The final output should be "
    "semantically correct and immediately usable.\n\n"
    "Do not include any meta-commentary about what you included or why. "
    "Focus solely on delivering the Markdown content."
)

# ============================================
# GLM-OCR Settings
# ============================================
# Uses HuggingFace Transformers for local inference.
# The model is loaded via AutoModelForImageTextToText with device_map="auto".

EMBED_SERVER_GLM_OCR_MODEL = 'MEMORYLAYER_EMBED_GLM_OCR_MODEL'
DEFAULT_EMBED_SERVER_GLM_OCR_MODEL = 'zai-org/GLM-OCR'

EMBED_SERVER_GLM_OCR_MAX_TOKENS = 'MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS'
DEFAULT_EMBED_SERVER_GLM_OCR_MAX_TOKENS = 16384

# ============================================
# DeepSeek-OCR-2 Settings
# ============================================
# Uses HuggingFace Transformers for local inference.
# The model is loaded via AutoModelForImageTextToText with device_map="auto".

EMBED_SERVER_DEEPSEEK_OCR_MODEL = 'MEMORYLAYER_EMBED_DEEPSEEK_OCR_MODEL'
DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MODEL = 'deepseek-ai/DeepSeek-OCR-2'

EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS = 'MEMORYLAYER_EMBED_DEEPSEEK_OCR_MAX_TOKENS'
DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS = 16384

# ============================================
# Gemini Fallback
# ============================================
EMBED_SERVER_GEMINI_MODEL = 'MEMORYLAYER_EMBED_GEMINI_MODEL'
DEFAULT_EMBED_SERVER_GEMINI_MODEL = 'gemini-3-flash-preview'

EMBED_SERVER_GEMINI_MAX_TOKENS = 'MEMORYLAYER_EMBED_GEMINI_MAX_TOKENS'
DEFAULT_EMBED_SERVER_GEMINI_MAX_TOKENS = 16384

# ============================================
# Single-Vector Embedding (vLLM)
# ============================================
# Reuses memorylayer_server config keys:
#   MEMORYLAYER_EMBEDDING_MODEL, MEMORYLAYER_EMBEDDING_DIMENSIONS
# And memorylayer_saas vLLM keys:
#   MEMORYLAYER_EMBEDDING_VLLM_DTYPE, MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH

# ============================================
# Multi-Vector Embedding (ColPali)
# ============================================
# ColPali config keys live in memorylayer_embed_server.services.embedding.colpali:
#   MEMORYLAYER_EMBEDDING_MODEL, MEMORYLAYER_EMBEDDING_DEVICE, MEMORYLAYER_EMBEDDING_REVISION

# ============================================
# Visual Tokenizer
# ============================================
# Visual-tokenizer config keys live in the optional enterprise overlay
# (memorylayer_embed_server_enterprise.config). The OSS server has no
# Qwen3.5 references; install the enterprise package to enable.

# ============================================
# Embedding Preload
# ============================================
EMBED_SERVER_PRELOAD_MODELS = 'MEMORYLAYER_EMBED_PRELOAD_MODELS'
DEFAULT_EMBED_SERVER_PRELOAD_MODELS = True

# ============================================
# Mock Providers (testing / lightweight integration)
# ============================================
# When set to "true", _setup_dual_embedding_service uses deterministic
# mock providers (numpy only, no torch) instead of vLLM + ColPali. The
# mock single- and multi-vector providers expose the same surface as
# the real ones, so /v1/embeddings, /v1/embeddings/multi,
# /v1/embeddings/images, and /v1/score all work end-to-end.
EMBED_SERVER_USE_MOCK_PROVIDERS = 'MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS'
DEFAULT_EMBED_SERVER_USE_MOCK_PROVIDERS = False

# When set to "true", the dual embedding service reuses the multi-vector
# provider (ColPali-shaped) as the single-vector provider, by mean-pooling
# its token vectors. Useful for heavy/test deployments that want a real
# GPU embedding path without also pulling in vLLM. Has no effect when
# EMBED_SERVER_USE_MOCK_PROVIDERS=true.
EMBED_SERVER_USE_MULTI_FOR_SINGLE = 'MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE'
DEFAULT_EMBED_SERVER_USE_MULTI_FOR_SINGLE = False

# Selects which provider serves single-vector requests. Multi-vector
# continues to use ColPali (or the mock for ``USE_MOCK_PROVIDERS``).
# Values:
#   ``vllm``     (default)  in-process vLLM with Qwen3-VL-Embedding-2B
#   ``openai``              HTTP call out to OpenAI (or OpenAI-compat endpoint
#                           — pair with MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL)
#   ``google``              HTTP call out to Google GenAI
#   ``colpali``             alias for EMBED_SERVER_USE_MULTI_FOR_SINGLE=true
#   ``mock``                deterministic numpy mock (no GPU, no network)
# USE_MOCK_PROVIDERS and USE_MULTI_FOR_SINGLE remain supported as
# higher-priority overrides for backwards compatibility.
EMBED_SERVER_SINGLE_VECTOR_PROVIDER = 'MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER'
DEFAULT_EMBED_SERVER_SINGLE_VECTOR_PROVIDER = 'vllm'

# ============================================
# LLM Inference Profiles (multi-profile vLLM subprocess)
# ============================================
# When enabled, the embed-server hosts one or more ``vllm serve`` child
# processes (each a chat LLM) and exposes ``POST /v1/chat/completions``,
# ``POST /v1/completions``, and ``GET /v1/models``. Routing is by the
# OpenAI-standard ``model`` field (matched against profile names, aliases,
# and served model names). Internal vLLM ports are auto-assigned from
# ``EMBED_SERVER_LLM_PORT_RANGE`` — operators only ever see the embed-server
# external port.
#
# Per-profile config is read from env vars of the shape:
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_MODEL
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_ALIASES         (comma list, optional)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_DTYPE           (default "auto")
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_MAX_MODEL_LEN   (optional int)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_GPU_MEM_UTIL    (default 0.25)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_ENFORCE_EAGER   (default false)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_TENSOR_PARALLEL_SIZE (default 1)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_STARTUP_TIMEOUT_SEC  (default 600)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_EXTRA_ARGS      (shell-split list)
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_CMD             (default "vllm")
#   MEMORYLAYER_EMBED_LLM_PROFILE_<NAME>_HOST            (default "127.0.0.1")
EMBED_SERVER_LLM_ENABLED = 'MEMORYLAYER_EMBED_LLM_ENABLED'
DEFAULT_EMBED_SERVER_LLM_ENABLED = False

EMBED_SERVER_LLM_PROFILES = 'MEMORYLAYER_EMBED_LLM_PROFILES'
DEFAULT_EMBED_SERVER_LLM_PROFILES = ''  # comma list of profile names

EMBED_SERVER_LLM_DEFAULT_PROFILE = 'MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE'
DEFAULT_EMBED_SERVER_LLM_DEFAULT_PROFILE = ''  # empty → no default; unknown model → 404

EMBED_SERVER_LLM_PRELOAD = 'MEMORYLAYER_EMBED_LLM_PRELOAD'
DEFAULT_EMBED_SERVER_LLM_PRELOAD = False

EMBED_SERVER_LLM_PORT_RANGE = 'MEMORYLAYER_EMBED_LLM_PORT_RANGE'
DEFAULT_EMBED_SERVER_LLM_PORT_RANGE = '18100-18199'

# Per-profile env-var prefix; the actual fields are read dynamically by
# ``services/llm/router.py`` so we don't need a constant per profile.
LLM_PROFILE_ENV_PREFIX = 'MEMORYLAYER_EMBED_LLM_PROFILE_'
