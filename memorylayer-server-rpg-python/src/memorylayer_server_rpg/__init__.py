# SPDX-License-Identifier: Apache-2.0
"""memorylayer-server RPG plugin package.

Provides Repository Planning Graph (RPG) services, API routes, and task
handlers as a third-party plugin to memorylayer-server. Discovered via the
``memorylayer_server.plugin_packages`` entry point group at server preconfigure
time.
"""

EXT_RPG_SERVICE = "memorylayer-rpg-service"
__version__ = "0.2.0"

__all__ = ["EXT_RPG_SERVICE", "__version__"]
