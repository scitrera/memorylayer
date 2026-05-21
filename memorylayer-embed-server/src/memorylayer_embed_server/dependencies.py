"""Dependency injection for MemoryLayer Embed Server.

Uses scitrera-app-framework plugin pattern for service initialization.
Wires transcription cascade and dual embedding service.
"""

import logging
from logging import Logger

from scitrera_app_framework import (
    Variables,
    async_plugins_ready,
    async_plugins_stopping,
    ext_parse_bool,
    get_logger,
    get_variables,
    init_framework_desktop,
)

from .config import (
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_ENABLED,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MODEL,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_TRANSPORT,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_CMD,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_PORT,
    DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_STARTUP_TIMEOUT_SEC,
    DEFAULT_EMBED_SERVER_GEMINI_ENABLED,
    DEFAULT_EMBED_SERVER_GEMINI_MAX_TOKENS,
    DEFAULT_EMBED_SERVER_GEMINI_MODEL,
    DEFAULT_EMBED_SERVER_GLM_OCR_ENABLED,
    DEFAULT_EMBED_SERVER_GLM_OCR_MAX_TOKENS,
    DEFAULT_EMBED_SERVER_GLM_OCR_MODEL,
    DEFAULT_EMBED_SERVER_GLM_OCR_TRANSPORT,
    DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_CMD,
    DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_GPU_MEM_UTIL,
    DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_PORT,
    DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_STARTUP_TIMEOUT_SEC,
    DEFAULT_EMBED_SERVER_LLM_DEFAULT_PROFILE,
    DEFAULT_EMBED_SERVER_LLM_ENABLED,
    DEFAULT_EMBED_SERVER_LLM_PORT_RANGE,
    DEFAULT_EMBED_SERVER_LLM_PROFILES,
    DEFAULT_EMBED_SERVER_MULTI_VECTOR_PROVIDER,
    DEFAULT_EMBED_SERVER_SINGLE_VECTOR_PROVIDER,
    DEFAULT_EMBED_SERVER_TRANSCRIPTION_ENABLED,
    DEFAULT_EMBED_SERVER_USE_MOCK_PROVIDERS,
    DEFAULT_EMBED_SERVER_USE_MULTI_FOR_SINGLE,
    EMBED_SERVER_DEEPSEEK_OCR_ENABLED,
    EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS,
    EMBED_SERVER_DEEPSEEK_OCR_MODEL,
    EMBED_SERVER_DEEPSEEK_OCR_TRANSPORT,
    EMBED_SERVER_DEEPSEEK_OCR_VLLM_CMD,
    EMBED_SERVER_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL,
    EMBED_SERVER_DEEPSEEK_OCR_VLLM_PORT,
    EMBED_SERVER_DEEPSEEK_OCR_VLLM_STARTUP_TIMEOUT_SEC,
    EMBED_SERVER_GEMINI_ENABLED,
    EMBED_SERVER_GEMINI_MAX_TOKENS,
    EMBED_SERVER_GEMINI_MODEL,
    EMBED_SERVER_GLM_OCR_ENABLED,
    EMBED_SERVER_GLM_OCR_MAX_TOKENS,
    EMBED_SERVER_GLM_OCR_MODEL,
    EMBED_SERVER_GLM_OCR_TRANSPORT,
    EMBED_SERVER_GLM_OCR_VLLM_CMD,
    EMBED_SERVER_GLM_OCR_VLLM_GPU_MEM_UTIL,
    EMBED_SERVER_GLM_OCR_VLLM_PORT,
    EMBED_SERVER_GLM_OCR_VLLM_STARTUP_TIMEOUT_SEC,
    EMBED_SERVER_LLM_DEFAULT_PROFILE,
    EMBED_SERVER_LLM_ENABLED,
    EMBED_SERVER_LLM_PORT_RANGE,
    EMBED_SERVER_LLM_PROFILES,
    EMBED_SERVER_MULTI_VECTOR_PROVIDER,
    EMBED_SERVER_SINGLE_VECTOR_PROVIDER,
    EMBED_SERVER_TRANSCRIPTION_ENABLED,
    EMBED_SERVER_USE_MOCK_PROVIDERS,
    EMBED_SERVER_USE_MULTI_FOR_SINGLE,
)


# noinspection PyTypeHints
def preconfigure(v: Variables = None) -> (Variables, dict):
    """Pre-configure the framework and register plugins."""
    from scitrera_app_framework import register_package_plugins

    from . import api, lifecycle, services  # noqa: F401

    # init framework
    v: Variables = init_framework_desktop(
        "memorylayer-embed-server",
        base_plugins=False,
        stateful_chdir=False,  # no stateful root needed (stateless server)
        async_auto_enabled=False,
        v=v,
    )

    # suppress noisy loggers
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)

    # avoid duplicate invocations
    if v.get("__preconfigure_complete__", default=False):
        return v, services

    logger = get_logger(v)

    # register plugins
    logger.debug("Registering embed server services")
    register_package_plugins(services.__package__, v, recursive=True)

    logger.debug("Registering lifecycle components")
    register_package_plugins(lifecycle.__package__, v, recursive=True)

    logger.debug("Registering API routes")
    register_package_plugins(api.__package__, v, recursive=True)

    # Reuse the OSS server's pluggable observability stack:
    #   - services/metrics: MetricsService abstraction + Noop/Prometheus plugins
    #   - lifecycle/otel:   OTel SDK init (TracerProvider + OTLP exporter)
    # Both are gated by their own env vars (MEMORYLAYER_METRICS_SERVICE,
    # MEMORYLAYER_OTEL_ENABLED) so the import is safe even when the
    # ``[observability]`` extra is not installed.
    try:
        register_package_plugins("memorylayer_server.services.metrics", v, recursive=True)
        logger.debug("Registered OSS metrics service plugins")
    except Exception as e:  # noqa: BLE001 - non-fatal
        logger.warning("Failed to register OSS metrics plugins: %s", e)
    try:
        from memorylayer_server.lifecycle import otel as _otel_mod  # noqa: F401

        register_package_plugins(_otel_mod.__name__, v, recursive=False)
        logger.debug("Registered OSS OTel init plugin")
    except Exception as e:  # noqa: BLE001 - non-fatal
        logger.warning("Failed to register OSS OTel plugin: %s", e)

    # Optional enterprise plugin overlay (e.g. visual-tokenizer). Discovered
    # via the standard plugin scanner so any add-on package that ships
    # plugins under its top-level namespace gets wired up automatically.
    try:
        import memorylayer_embed_server_enterprise as _ent_pkg

        register_package_plugins(_ent_pkg.__name__, v, recursive=True)
        logger.info("Loaded enterprise embed-server extensions from %s", _ent_pkg.__name__)
    except ImportError:
        logger.debug("memorylayer_embed_server_enterprise not installed; skipping")

    v.set("__preconfigure_complete__", True)
    return v, services


async def initialize_services(v: Variables = None) -> Variables:
    """Initialize all services on application startup."""

    v, services = preconfigure(v)
    logger = get_logger(v)

    logger.debug("Initializing services")
    from scitrera_app_framework.core.plugins import init_all_plugins

    init_all_plugins(v, async_enabled=False)
    await async_plugins_ready(v)

    # Wire up transcription cascade
    transcription_enabled = v.environ(
        EMBED_SERVER_TRANSCRIPTION_ENABLED,
        default=DEFAULT_EMBED_SERVER_TRANSCRIPTION_ENABLED,
        type_fn=ext_parse_bool,
    )

    if transcription_enabled:
        logger.info("Setting up transcription cascade")
        _setup_transcription_cascade(v, logger)

    # Wire up dual embedding service
    logger.info("Setting up dual embedding service")
    _setup_dual_embedding_service(v, logger)

    # Visual tokenizer is registered from the optional enterprise overlay
    # (memorylayer-embed-server-enterprise) via the standard plugin scanner;
    # nothing wired here in OSS.

    # Wire up the LLM routing service (multi-profile vLLM subprocesses).
    # Gated on MEMORYLAYER_EMBED_LLM_ENABLED so existing deployments are
    # byte-for-byte unaffected by this feature when they don't opt in.
    llm_enabled = v.environ(
        EMBED_SERVER_LLM_ENABLED,
        default=DEFAULT_EMBED_SERVER_LLM_ENABLED,
        type_fn=ext_parse_bool,
    )
    if llm_enabled:
        logger.info("Setting up LLM routing service")
        _setup_llm_service(v, logger)

    # Set up GPU monitor
    from .services.gpu import GPUStatusMonitor

    gpu_monitor = GPUStatusMonitor(v=v)
    v.set("gpu_monitor", gpu_monitor)
    logger.info("GPU monitor initialized")

    return v


def _setup_transcription_cascade(v: Variables, logger: Logger):
    """Wire up the transcription cascade with configured providers.

    Each provider is gated on its own enable flag so deployments and tests
    can isolate a single cascade member. All default to true — existing
    deployments behave identically.
    """
    from .services.transcription.cascade import CascadeTranscriber

    providers = []

    # GLM-OCR (primary)
    if v.environ(EMBED_SERVER_GLM_OCR_ENABLED, default=DEFAULT_EMBED_SERVER_GLM_OCR_ENABLED, type_fn=ext_parse_bool):
        glm_provider = _init_glm_ocr_provider(v, logger)
        if glm_provider is not None:
            providers.append(glm_provider)
    else:
        logger.info("GLM-OCR provider disabled by MEMORYLAYER_EMBED_GLM_OCR_ENABLED=false")

    # DeepSeek-OCR-2 (secondary)
    if v.environ(EMBED_SERVER_DEEPSEEK_OCR_ENABLED, default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_ENABLED, type_fn=ext_parse_bool):
        deepseek_provider = _init_deepseek_ocr_provider(v, logger)
        if deepseek_provider is not None:
            providers.append(deepseek_provider)
    else:
        logger.info("DeepSeek-OCR-2 provider disabled by MEMORYLAYER_EMBED_DEEPSEEK_OCR_ENABLED=false")

    # Gemini Flash (fallback - external API)
    if v.environ(EMBED_SERVER_GEMINI_ENABLED, default=DEFAULT_EMBED_SERVER_GEMINI_ENABLED, type_fn=ext_parse_bool):
        try:
            from .services.transcription.gemini import GeminiProvider

            gemini_provider = GeminiProvider(
                v=v,
                model_name=v.environ(EMBED_SERVER_GEMINI_MODEL, default=DEFAULT_EMBED_SERVER_GEMINI_MODEL),
                max_tokens=v.environ(EMBED_SERVER_GEMINI_MAX_TOKENS, default=DEFAULT_EMBED_SERVER_GEMINI_MAX_TOKENS, type_fn=int),
            )
            providers.append(gemini_provider)
            logger.info("Gemini provider configured")
        except ImportError as e:
            logger.warning("Gemini provider unavailable (google-genai not installed): %s", e)
    else:
        logger.info("Gemini provider disabled by MEMORYLAYER_EMBED_GEMINI_ENABLED=false")

    cascade = CascadeTranscriber(v=v, providers=providers)
    v.set("cascade_transcriber", cascade)
    logger.info("Transcription cascade configured with %d providers", len(providers))


def _setup_dual_embedding_service(v: Variables, logger: Logger):
    """Wire up the dual embedding service with both providers."""
    from .services.embedding.dual_service import DualEmbeddingService

    use_mocks = v.environ(
        EMBED_SERVER_USE_MOCK_PROVIDERS,
        default=DEFAULT_EMBED_SERVER_USE_MOCK_PROVIDERS,
        type_fn=ext_parse_bool,
    )
    use_multi_for_single = v.environ(
        EMBED_SERVER_USE_MULTI_FOR_SINGLE,
        default=DEFAULT_EMBED_SERVER_USE_MULTI_FOR_SINGLE,
        type_fn=ext_parse_bool,
    )
    single_provider_kind = (
        v.environ(
            EMBED_SERVER_SINGLE_VECTOR_PROVIDER,
            default=DEFAULT_EMBED_SERVER_SINGLE_VECTOR_PROVIDER,
        )
        .strip()
        .lower()
    )
    # ``colpali`` is sugar for the multi-for-single flag — fold them.
    if single_provider_kind == "colpali":
        use_multi_for_single = True

    single_vector = None
    multi_vector = None

    if use_mocks:
        # Deterministic numpy-only providers — no torch, no model downloads.
        # See services/embedding/mock_providers.py for the rationale.
        from .services.embedding.mock_providers import (
            MockMultiVectorProvider,
            MockSingleVectorProvider,
        )

        single_vector = MockSingleVectorProvider(v=v)
        multi_vector = MockMultiVectorProvider(v=v)
        logger.info("Mock embedding providers configured (EMBED_SERVER_USE_MOCK_PROVIDERS=true)")
    else:
        # Single-vector: pick provider by EMBED_SERVER_SINGLE_VECTOR_PROVIDER.
        # ``multi-for-single`` (or USE_MULTI_FOR_SINGLE=true above) skips
        # the single-vector init entirely and the shared-provider block
        # below wires ColPali in.
        if not use_multi_for_single:
            single_vector = _init_single_vector_provider(v, logger, single_provider_kind)

        # Multi-vector: dispatch on EMBED_SERVER_MULTI_VECTOR_PROVIDER.
        # Both backends speak the same MultimodalEmbeddingProvider surface,
        # so /v1/embeddings/multi, /v1/embeddings/images, and /v1/score are
        # backend-agnostic on the wire.
        multi_provider_kind = (
            v.environ(
                EMBED_SERVER_MULTI_VECTOR_PROVIDER,
                default=DEFAULT_EMBED_SERVER_MULTI_VECTOR_PROVIDER,
            )
            .strip()
            .lower()
        )
        multi_vector = _init_multi_vector_provider(v, logger, multi_provider_kind)

    # Optional: share the multi-vector provider as the single-vector provider
    # so a real-ColPali deployment doesn't also need vLLM installed. ColPali
    # exposes ``.embed()`` (mean-pooled from its multi-vector output).
    if use_multi_for_single and single_vector is None and multi_vector is not None:
        single_vector = multi_vector
        logger.info("EMBED_SERVER_USE_MULTI_FOR_SINGLE=true: reusing multi-vector provider as single-vector (mean-pooled)")

    dual_service = DualEmbeddingService(
        v=v,
        single_vector_provider=single_vector,
        multi_vector_provider=multi_vector,
    )
    v.set("dual_embedding_service", dual_service)
    logger.info("Dual embedding service configured")


def _init_single_vector_provider(v: Variables, logger: Logger, kind: str):
    """Construct the single-vector embedding provider chosen by ``kind``.

    Returns ``None`` on import / init failure so the caller can decide
    whether to fall back. Each branch reuses the upstream OSS
    EmbeddingProviderPlugin classes (vLLM lives in this package; OpenAI
    and Google live in memorylayer-server) and bypasses framework-level
    plugin selection — the kind here is the dispatch key.
    """
    if kind in ("vllm", "", "default"):
        try:
            from memorylayer_embed_server.services.embedding.vllm import VLLMEmbeddingProviderPlugin

            provider = VLLMEmbeddingProviderPlugin().initialize(v, logger)
            logger.info("Single-vector (vLLM in-process) provider configured")
            return provider
        except ImportError as e:
            logger.warning("vLLM embedding provider unavailable: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize vLLM embedding provider: %s", e)
        return None

    if kind == "vllm_subprocess":
        try:
            from memorylayer_embed_server.services.embedding.vllm_subprocess import (
                VLLMSubprocessEmbeddingProviderPlugin,
            )

            provider = VLLMSubprocessEmbeddingProviderPlugin().initialize(v, logger)
            logger.info("Single-vector (vLLM subprocess) provider configured")
            return provider
        except ImportError as e:
            logger.warning("vLLM subprocess provider unavailable: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize vLLM subprocess provider: %s", e)
        return None

    if kind == "openai":
        try:
            from memorylayer_server.services.embedding.openai import OpenAIEmbeddingProviderPlugin

            provider = OpenAIEmbeddingProviderPlugin().initialize(v, logger)
            logger.info("Single-vector (OpenAI / OpenAI-compat) provider configured")
            return provider
        except ImportError as e:
            logger.warning(
                "OpenAI embedding provider unavailable (install the `openai` extra on memorylayer-server): %s",
                e,
            )
        except Exception as e:
            logger.warning("Failed to initialize OpenAI embedding provider: %s", e)
        return None

    if kind == "google":
        try:
            from memorylayer_server.services.embedding.google import GoogleEmbeddingProviderPlugin

            provider = GoogleEmbeddingProviderPlugin().initialize(v, logger)
            logger.info("Single-vector (Google GenAI) provider configured")
            return provider
        except ImportError as e:
            logger.warning(
                "Google embedding provider unavailable (install the `google` extra on memorylayer-server): %s",
                e,
            )
        except Exception as e:
            logger.warning("Failed to initialize Google embedding provider: %s", e)
        return None

    if kind == "mock":
        from .services.embedding.mock_providers import MockSingleVectorProvider

        logger.info("Single-vector (mock) provider configured")
        return MockSingleVectorProvider(v=v)

    logger.warning(
        "Unknown EMBED_SERVER_SINGLE_VECTOR_PROVIDER value: %r — "
        "no single-vector provider will be configured. Valid values: "
        "vllm, vllm_subprocess, openai, google, colpali, mock.",
        kind,
    )
    return None


def _init_multi_vector_provider(v: Variables, logger: Logger, kind: str):
    """Construct the multi-vector / ColPali embedding provider for ``kind``.

    ``colpali_inprocess`` runs colpali-engine in-process via HF transformers
    — small, simple, good for tests and tiny-model deployments. ``vllm_subprocess``
    (production default) spawns ``vllm serve --runner pooling`` for batched
    throughput. Both providers expose the same MultimodalEmbeddingProvider
    surface so the route layer is unchanged. The naming mirrors the
    single-vector dispatcher (``vllm`` in-process, ``vllm_subprocess``).
    """
    if kind in ("", "colpali_inprocess", "colpali"):
        try:
            from memorylayer_embed_server.services.embedding.colpali import ColPaliEmbeddingProviderPlugin

            provider = ColPaliEmbeddingProviderPlugin().initialize(v, logger)
            logger.info("Multi-vector (ColPali in-process via colpali-engine) provider configured")
            return provider
        except ImportError as e:
            logger.warning("ColPali in-process provider unavailable: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize ColPali in-process provider: %s", e)
        return None

    if kind == "vllm_subprocess":
        try:
            from memorylayer_embed_server.services.embedding.vllm_multi_vector import (
                VLLMMultiVectorProviderPlugin,
            )

            provider = VLLMMultiVectorProviderPlugin().initialize(v, logger)
            logger.info("Multi-vector (vLLM subprocess --runner pooling) provider configured")
            return provider
        except ImportError as e:
            logger.warning("vLLM multi-vector provider unavailable: %s", e)
        except Exception as e:
            logger.warning("Failed to initialize vLLM multi-vector provider: %s", e)
        return None

    logger.warning(
        "Unknown EMBED_SERVER_MULTI_VECTOR_PROVIDER value: %r — no multi-vector "
        "provider will be configured. Valid values: colpali_inprocess, vllm_subprocess.",
        kind,
    )
    return None


def _init_glm_ocr_provider(v: Variables, logger: Logger):
    """Construct the GLM-OCR transcription provider per the configured transport.

    ``vllm_subprocess`` (default) runs the upstream recipe with MTP
    speculative decoding; ``hf`` keeps the legacy in-process HF path.
    """
    transport = (
        v.environ(EMBED_SERVER_GLM_OCR_TRANSPORT, default=DEFAULT_EMBED_SERVER_GLM_OCR_TRANSPORT)
        .strip()
        .lower()
    )
    model_name = v.environ(EMBED_SERVER_GLM_OCR_MODEL, default=DEFAULT_EMBED_SERVER_GLM_OCR_MODEL)
    max_tokens = v.environ(EMBED_SERVER_GLM_OCR_MAX_TOKENS, default=DEFAULT_EMBED_SERVER_GLM_OCR_MAX_TOKENS, type_fn=int)

    if transport == "vllm_subprocess":
        try:
            from .services.transcription.vllm_transcription import build_glm_ocr_vllm_provider

            provider = build_glm_ocr_vllm_provider(
                v=v,
                logger=logger,
                model_name=model_name,
                max_tokens=max_tokens,
                port=v.environ(EMBED_SERVER_GLM_OCR_VLLM_PORT, default=DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_PORT, type_fn=int),
                gpu_memory_utilization=v.environ(
                    EMBED_SERVER_GLM_OCR_VLLM_GPU_MEM_UTIL,
                    default=DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_GPU_MEM_UTIL,
                    type_fn=float,
                ),
                startup_timeout_sec=v.environ(
                    EMBED_SERVER_GLM_OCR_VLLM_STARTUP_TIMEOUT_SEC,
                    default=DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_STARTUP_TIMEOUT_SEC,
                    type_fn=float,
                ),
                cmd=v.environ(EMBED_SERVER_GLM_OCR_VLLM_CMD, default=DEFAULT_EMBED_SERVER_GLM_OCR_VLLM_CMD),
            )
            logger.info("GLM-OCR provider configured (transport=vllm_subprocess)")
            return provider
        except ImportError as e:
            logger.warning("GLM-OCR vllm-subprocess provider unavailable: %s", e)
            return None

    if transport == "hf":
        try:
            from .services.transcription.glm_ocr import GLMOCRProvider

            provider = GLMOCRProvider(v=v, model_name=model_name, max_tokens=max_tokens)
            logger.info("GLM-OCR provider configured (transport=hf)")
            return provider
        except ImportError as e:
            logger.warning("GLM-OCR HF provider unavailable (transformers not installed): %s", e)
            return None

    logger.warning(
        "Unknown MEMORYLAYER_EMBED_GLM_OCR_TRANSPORT value: %r — GLM-OCR will not "
        "be configured. Valid values: vllm_subprocess, hf.",
        transport,
    )
    return None


def _init_deepseek_ocr_provider(v: Variables, logger: Logger):
    """Construct the DeepSeek-OCR-2 transcription provider per the configured transport.

    ``vllm_subprocess`` (default) runs the upstream recipe with the
    DeepSeek-specific logits processor and prefix-caching/MM-cache off;
    ``hf`` keeps the legacy in-process HF path (currently only works
    with eager attention, so significantly slower).
    """
    transport = (
        v.environ(EMBED_SERVER_DEEPSEEK_OCR_TRANSPORT, default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_TRANSPORT)
        .strip()
        .lower()
    )
    model_name = v.environ(EMBED_SERVER_DEEPSEEK_OCR_MODEL, default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MODEL)
    max_tokens = v.environ(
        EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS, default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS, type_fn=int
    )

    if transport == "vllm_subprocess":
        try:
            from .services.transcription.vllm_transcription import build_deepseek_ocr_vllm_provider

            provider = build_deepseek_ocr_vllm_provider(
                v=v,
                logger=logger,
                model_name=model_name,
                max_tokens=max_tokens,
                port=v.environ(
                    EMBED_SERVER_DEEPSEEK_OCR_VLLM_PORT,
                    default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_PORT,
                    type_fn=int,
                ),
                gpu_memory_utilization=v.environ(
                    EMBED_SERVER_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL,
                    default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL,
                    type_fn=float,
                ),
                startup_timeout_sec=v.environ(
                    EMBED_SERVER_DEEPSEEK_OCR_VLLM_STARTUP_TIMEOUT_SEC,
                    default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_STARTUP_TIMEOUT_SEC,
                    type_fn=float,
                ),
                cmd=v.environ(EMBED_SERVER_DEEPSEEK_OCR_VLLM_CMD, default=DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_VLLM_CMD),
            )
            logger.info("DeepSeek-OCR-2 provider configured (transport=vllm_subprocess)")
            return provider
        except ImportError as e:
            logger.warning("DeepSeek-OCR-2 vllm-subprocess provider unavailable: %s", e)
            return None

    if transport == "hf":
        try:
            from .services.transcription.deepseek_ocr import DeepSeekOCRProvider

            provider = DeepSeekOCRProvider(v=v, model_name=model_name, max_tokens=max_tokens)
            logger.info("DeepSeek-OCR-2 provider configured (transport=hf)")
            return provider
        except ImportError as e:
            logger.warning("DeepSeek-OCR-2 HF provider unavailable (transformers not installed): %s", e)
            return None

    logger.warning(
        "Unknown MEMORYLAYER_EMBED_DEEPSEEK_OCR_TRANSPORT value: %r — DeepSeek-OCR-2 "
        "will not be configured. Valid values: vllm_subprocess, hf.",
        transport,
    )
    return None


def _setup_llm_service(v: Variables, logger: Logger):
    """Discover LLM profiles from env vars and assemble the routing service.

    Each profile gets its own free loopback port (from the configured port
    range) so operators never have to set ports manually. Failures on
    individual profiles are logged but don't abort wiring of siblings —
    the routing service simply skips the failed profile.
    """
    from .services._vllm_runner import find_free_port
    from .services.llm.router import LLMRoutingService
    from .services.llm.vllm_subprocess import build_provider_from_env

    profiles_env = v.environ(
        EMBED_SERVER_LLM_PROFILES,
        default=DEFAULT_EMBED_SERVER_LLM_PROFILES,
    )
    profile_names = [p.strip() for p in profiles_env.split(",") if p.strip()]
    if not profile_names:
        logger.warning(
            "MEMORYLAYER_EMBED_LLM_ENABLED=true but MEMORYLAYER_EMBED_LLM_PROFILES is empty — no LLM profiles will be configured."
        )
        return

    # Parse the port range; we allocate ports inside it so they cluster
    # together for ops + firewall reviews.
    port_range_str = v.environ(
        EMBED_SERVER_LLM_PORT_RANGE,
        default=DEFAULT_EMBED_SERVER_LLM_PORT_RANGE,
    )
    try:
        low_str, high_str = port_range_str.split("-", 1)
        port_low, port_high = int(low_str), int(high_str)
    except (ValueError, AttributeError):
        logger.warning(
            "Invalid MEMORYLAYER_EMBED_LLM_PORT_RANGE=%r; falling back to %r",
            port_range_str,
            DEFAULT_EMBED_SERVER_LLM_PORT_RANGE,
        )
        low_str, high_str = DEFAULT_EMBED_SERVER_LLM_PORT_RANGE.split("-", 1)
        port_low, port_high = int(low_str), int(high_str)

    profiles: dict = {}
    used_ports: set[int] = set()
    for name in profile_names:
        try:
            # Allocate inside the configured range, skipping ports already
            # taken by sibling profiles we just allocated this turn.
            port = None
            for _ in range(port_high - port_low + 1):
                candidate = find_free_port(low=port_low, high=port_high)
                if candidate not in used_ports:
                    port = candidate
                    break
                # Try again — find_free_port might keep returning the same
                # port if siblings haven't bound yet; bumping low avoids
                # an infinite loop.
                port_low = max(port_low + 1, candidate + 1)
            if port is None:
                raise RuntimeError(f"could not find free port in range {port_range_str} for profile {name}")
            used_ports.add(port)
            provider = build_provider_from_env(v, logger, profile_name=name, port=port)
            profiles[name] = provider
            logger.info(
                "LLM profile configured: name=%s, model=%s, port=%d, aliases=%s",
                name,
                provider.model_name,
                port,
                provider.aliases,
            )
        except Exception as e:  # noqa: BLE001 - log and skip the bad profile
            logger.warning("Failed to configure LLM profile %r: %s", name, e)

    if not profiles:
        logger.warning("No LLM profiles were configured successfully")
        return

    default_profile = (
        v.environ(
            EMBED_SERVER_LLM_DEFAULT_PROFILE,
            default=DEFAULT_EMBED_SERVER_LLM_DEFAULT_PROFILE,
        ).strip()
        or None
    )
    if default_profile and default_profile not in profiles:
        logger.warning(
            "MEMORYLAYER_EMBED_LLM_DEFAULT_PROFILE=%r is not in declared profiles %s; default-fallback will be disabled.",
            default_profile,
            list(profiles),
        )
        default_profile = None

    svc = LLMRoutingService(
        profiles=profiles,
        default_profile=default_profile,
        logger=logger,
    )
    v.set("llm_routing_service", svc)
    logger.info(
        "LLM routing service configured: profiles=%s, default=%r",
        list(profiles),
        default_profile,
    )


async def shutdown_services(v: Variables = None) -> None:
    """Shutdown all services on application shutdown."""

    v = get_variables(v)
    logger = get_logger(v)

    logger.debug("Shutting down embed server services")

    # Shut LLM subprocesses down first — they hold GPU memory and child
    # processes that need explicit teardown (process-group SIGTERM/KILL).
    llm_svc = v.get("llm_routing_service", default=None)
    if llm_svc is not None:
        try:
            await llm_svc.shutdown()
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            logger.warning("LLM routing service shutdown failed: %s", e)

    # If a vLLM subprocess was started by the embed-server, stop it before
    # the framework tears the rest of the plugins down so we don't leak
    # GPU memory or orphan a child process.
    dual_service = v.get("dual_embedding_service", default=None)
    if dual_service is not None:
        single = getattr(dual_service, "_single_vector", None)
        if single is not None and hasattr(single, "shutdown") and getattr(single, "_process", None) is not None:
            try:
                await single.shutdown()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning("vllm subprocess shutdown failed: %s", e)

    await async_plugins_stopping(v)

    from scitrera_app_framework.core.plugins import shutdown_all_plugins

    shutdown_all_plugins(v)
