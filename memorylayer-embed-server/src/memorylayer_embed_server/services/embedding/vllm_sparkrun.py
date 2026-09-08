"""Single-vector provider whose vLLM process is launched by sparkrun.

Identical to ``vllm_subprocess`` above the process boundary — same OpenAI-compatible
client, same batching, same dimension handling — differing only in who starts and
stops ``vllm serve``. The launch parameters come from a recipe shipped with the
package rather than being composed in Python.

Why: those parameters already exist as sparkrun recipes that run these exact models,
and were duplicated here as ``MEMORYLAYER_EMBEDDING_VLLM_*`` env vars. The recipe is
the more expressive form and is shared with the rest of the stack.

    MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_sparkrun

Requires the ``sparkrun`` extra:

    pip install "memorylayer-embed-server[sparkrun]"

The same env vars as ``vllm_subprocess`` still apply — they are threaded onto the
recipe as launch overrides, so the two providers are configured identically and can
be swapped without touching deployment config. Set
``MEMORYLAYER_EMBEDDING_SPARKRUN_RECIPE`` to use a recipe of your own instead of the
packaged one.
"""

from logging import Logger

from memorylayer_server.config import MEMORYLAYER_EMBEDDING_DIMENSIONS, MEMORYLAYER_EMBEDDING_MODEL
from memorylayer_server.services.embedding.base import EmbeddingProviderPluginBase
from scitrera_app_framework import Variables, ext_parse_bool

from .._sparkrun_runner import SparkrunVLLMRunner, default_recipe_path
from .vllm_subprocess import (
    DEFAULT_DTYPE,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENFORCE_EAGER,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_LENGTH,
    DEFAULT_VLLM_SUBPROCESS_HOST,
    DEFAULT_VLLM_SUBPROCESS_PORT,
    DEFAULT_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
    MEMORYLAYER_EMBEDDING_VLLM_DTYPE,
    MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER,
    MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL,
    MEMORYLAYER_EMBEDDING_VLLM_MAX_CONCURRENT,
    MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH,
    MEMORYLAYER_EMBEDDING_VLLM_MODEL,
    MEMORYLAYER_EMBEDDING_VLLM_OVERSUBSCRIBE,
    MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST,
    MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT,
    MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
    VLLMSubprocessEmbeddingProvider,
    _parse_max_concurrent_env,
    _parse_oversubscribe_factor_env,
)

PROVIDER_NAME_VLLM_SPARKRUN = "vllm_sparkrun"

MEMORYLAYER_EMBEDDING_SPARKRUN_RECIPE = "MEMORYLAYER_EMBEDDING_SPARKRUN_RECIPE"
DEFAULT_SPARKRUN_RECIPE = "sv_qwen3vl_2b.yaml"

MEMORYLAYER_EMBEDDING_SPARKRUN_REUSE = "MEMORYLAYER_EMBEDDING_SPARKRUN_REUSE"
DEFAULT_SPARKRUN_REUSE = True

__all__ = ["VLLMSparkrunEmbeddingProviderPlugin", "PROVIDER_NAME_VLLM_SPARKRUN"]


class VLLMSparkrunEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    """Plugin entry point; dispatched from ``_init_single_vector_provider``."""

    PROVIDER_NAME = PROVIDER_NAME_VLLM_SPARKRUN

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        model_name = v.environ(MEMORYLAYER_EMBEDDING_VLLM_MODEL, default=None)
        if not model_name:
            model_name = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL)

        recipe = v.environ(MEMORYLAYER_EMBEDDING_SPARKRUN_RECIPE, default=None)
        recipe_path = recipe if recipe else default_recipe_path(DEFAULT_SPARKRUN_RECIPE)

        runner = SparkrunVLLMRunner(
            recipe_path=recipe_path,
            reuse_existing=v.environ(
                MEMORYLAYER_EMBEDDING_SPARKRUN_REUSE,
                default=DEFAULT_SPARKRUN_REUSE,
                type_fn=ext_parse_bool,
            ),
            # Everything below mirrors the vllm_subprocess plugin so the two
            # providers read the same configuration.
            role="embedding",
            model_name=model_name,
            host=v.environ(MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST, default=DEFAULT_VLLM_SUBPROCESS_HOST),
            port=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT,
                default=DEFAULT_VLLM_SUBPROCESS_PORT,
                type_fn=int,
            ),
            dtype=v.environ(MEMORYLAYER_EMBEDDING_VLLM_DTYPE, default=DEFAULT_DTYPE),
            max_model_len=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH,
                default=DEFAULT_MAX_LENGTH,
                type_fn=int,
            ),
            gpu_memory_utilization=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL,
                default=DEFAULT_GPU_MEMORY_UTILIZATION,
                type_fn=float,
            ),
            enforce_eager=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER,
                default=DEFAULT_ENFORCE_EAGER,
                type_fn=ext_parse_bool,
            ),
            startup_timeout_sec=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
                default=DEFAULT_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
                type_fn=float,
            ),
            max_concurrent=_parse_max_concurrent_env(v.environ(MEMORYLAYER_EMBEDDING_VLLM_MAX_CONCURRENT, default=None)),
            oversubscribe_factor=_parse_oversubscribe_factor_env(
                v.environ(MEMORYLAYER_EMBEDDING_VLLM_OVERSUBSCRIBE, default=None)
            ),
            logger=logger,
        )

        return VLLMSubprocessEmbeddingProvider(
            v=v,
            model_name=model_name,
            output_dimensions=v.environ(
                MEMORYLAYER_EMBEDDING_DIMENSIONS,
                default=DEFAULT_EMBEDDING_DIMENSIONS,
                type_fn=int,
            ),
            runner=runner,
        )
