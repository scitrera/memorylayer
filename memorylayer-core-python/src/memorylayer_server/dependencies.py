"""dependency injection for MemoryLayer.ai.

Uses scitrera-app-framework plugin pattern for service initialization.
Services are lazily initialized on first access via get_extension().
"""
import logging
from logging import Logger
from typing import Callable

from scitrera_app_framework import (
    Variables, get_variables, get_logger, init_framework_desktop,
    async_plugins_ready, async_plugins_stopping
)
from .config import MEMORYLAYER_DATA_DIR

# global preconfigure hooks (not specific to variables instance)
_preconfigure_hooks: list[Callable[[Variables], None]] = []


# noinspection PyTypeHints
def preconfigure(v: Variables = None, test_mode: bool = False, test_logger: Logger = None) -> (Variables, dict):
    """ Pre-configure the framework """
    from scitrera_app_framework import register_package_plugins
    from . import api, services, lifecycle  # noqa: F401

    # handle test mode
    additional_kwargs = {} if not test_mode else {
        'fault_handler': False,
        'fixed_logger': test_logger,
        'pyroscope': False,
        'shutdown_hooks': False,
    }

    # init framework (has internal protection against multiple invocations)
    v: Variables = init_framework_desktop(
        'memorylayer-server',
        base_plugins=False,  # disable base plugins (we don't need them)
        stateful_chdir=True,  # change working directory to stateful root
        stateful_root_env_key=MEMORYLAYER_DATA_DIR,  # use MEMORYLAYER_DATA_DIR for stateful root
        async_auto_enabled=False,  # manage async plugin lifecycle hooks manually
        v=v,  # allow variables instance pass-through
        **additional_kwargs
    )

    # do some custom logging tweaks (TODO: upstream mechanism for logging tweaks to scitrera-app-framework)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('aiosqlite').setLevel(logging.WARNING)
    logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
    logging.getLogger('httpcore.connection').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)

    # avoid duplicate invocations of preconfigure()
    if v.get('__preconfigure_complete__', default=False):
        return v, services

    logger = get_logger(v)

    # register plugins
    logger.debug('Registering core services')
    register_package_plugins(services.__package__, v, recursive=True)

    logger.debug('Registering lifecycle components')
    register_package_plugins(lifecycle.__package__, v, recursive=True)

    logger.debug('Registering API Routes')
    register_package_plugins(api.__package__, v, recursive=True)

    # handle preconfiguration hooks
    logger.debug('Evaluating preconfigure hooks')
    global _preconfigure_hooks
    if v.get('__preconfigure_hooks_installed__', default=0) == (lph := len(_preconfigure_hooks)):
        return v, services

    # run through preconfigure hooks (allows for registering additional plugins before initialization)
    for hook in _preconfigure_hooks:
        hook(v)

    v.set('__preconfigure_hooks_installed__', lph)
    logger.debug('Installed preconfiguration hooks')

    v.set('__preconfigure_complete__', True)
    return v, services


async def initialize_services(v: Variables = None) -> Variables:
    """Initialize all services on application startup."""

    # ensure preconfigured
    v, services = preconfigure(v)
    logger = get_logger(v)

    logger.debug("Initializing services")
    from scitrera_app_framework.core.plugins import init_all_plugins
    init_all_plugins(v, async_enabled=False)  # handle sync part
    await async_plugins_ready(v)  # handle async part with sequencing managed

    return v


async def shutdown_services(v: Variables = None) -> None:
    """Shutdown all services on application shutdown."""

    v = get_variables(v)
    logger = get_logger(v)

    logger.debug("Shutting down services")
    await async_plugins_stopping(v)

    from scitrera_app_framework.core.plugins import shutdown_all_plugins
    shutdown_all_plugins(v)
