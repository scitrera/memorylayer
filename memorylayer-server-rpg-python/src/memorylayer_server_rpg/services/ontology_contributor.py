# SPDX-License-Identifier: Apache-2.0
"""RPG ontology contributor — contributes code_structure relationship types at startup.

The 12 code_structure relationship types are the single source of truth for the
RPG plugin. ``RpgOntologyContributorPlugin`` is discovered automatically via the
``EXT_MULTI_ONTOLOGY_CONTRIBUTORS`` extension point and merged into the live
ontology service during its ``async_ready`` hook.
"""

import logging

from memorylayer_server.services.ontology.contributor import OntologyContributorPlugin

logger = logging.getLogger(__name__)

# Single source of truth for all RPG code-structure relationship types.
RPG_RELATIONSHIP_TYPES: dict[str, dict] = {
    "contains": {
        "description": "A contains B (directory contains file, file contains class, etc.)",
        "symmetric": False,
        "transitive": True,
        "inverse": "contained_by",
        "category": "code_structure",
    },
    "contained_by": {
        "description": "A is contained within B",
        "symmetric": False,
        "transitive": True,
        "inverse": "contains",
        "category": "code_structure",
    },
    "inherits": {
        "description": "A inherits from B (class inheritance)",
        "symmetric": False,
        "transitive": True,
        "inverse": "inherited_by",
        "category": "code_structure",
    },
    "inherited_by": {
        "description": "A is inherited by B",
        "symmetric": False,
        "transitive": True,
        "inverse": "inherits",
        "category": "code_structure",
    },
    "invokes": {
        "description": "A invokes/calls B (function/method call)",
        "symmetric": False,
        "transitive": False,
        "inverse": "invoked_by",
        "category": "code_structure",
    },
    "invoked_by": {
        "description": "A is invoked/called by B",
        "symmetric": False,
        "transitive": False,
        "inverse": "invokes",
        "category": "code_structure",
    },
    "imports": {
        "description": "A imports B (module/package import)",
        "symmetric": False,
        "transitive": False,
        "inverse": "imported_by",
        "category": "code_structure",
    },
    "imported_by": {
        "description": "A is imported by B",
        "symmetric": False,
        "transitive": False,
        "inverse": "imports",
        "category": "code_structure",
    },
    "composes": {
        "description": "A composes B (composition/aggregation relationship)",
        "symmetric": False,
        "transitive": False,
        "inverse": "composed_by",
        "category": "code_structure",
    },
    "composed_by": {
        "description": "A is composed by B",
        "symmetric": False,
        "transitive": False,
        "inverse": "composes",
        "category": "code_structure",
    },
    "data_flow": {
        "description": "Data flows from A to B",
        "symmetric": False,
        "transitive": False,
        "inverse": "data_flow_from",
        "category": "code_structure",
    },
    "data_flow_from": {
        "description": "A receives data flow from B",
        "symmetric": False,
        "transitive": False,
        "inverse": "data_flow",
        "category": "code_structure",
    },
}


# Single source of truth for the 9 RPG memory subtypes. Stored as a
# {memory_type: {subtypes}} mapping to mirror OSS_KNOWN_SUBTYPES. RPG nodes
# are persisted as Memory entities with type=SEMANTIC, so all subtypes are
# scoped under "semantic".
RPG_SUBTYPES: dict[str, set[str]] = {
    "semantic": {
        "rpg_directory",
        "rpg_file",
        "rpg_class",
        "rpg_function",
        "rpg_method",
        "rpg_component",
        "rpg_module",
        "rpg_package",
        "rpg_interface",
    },
}


class RpgOntologyContributorPlugin(OntologyContributorPlugin):
    """Contributes the 12 code_structure relationship types to the ontology service.

    Discovered automatically via the ``EXT_MULTI_ONTOLOGY_CONTRIBUTORS``
    extension point. The ontology service merges these types during its
    ``async_ready`` hook before any request-time ontology queries are served.

    This plugin is the sole source of RPG ``code_structure`` relationship
    types and RPG memory subtypes; the OSS core remains feature-agnostic.
    """

    def get_relationship_types(self) -> dict[str, dict]:
        """Return all 12 RPG code-structure relationship types."""
        logger.debug(
            "RpgOntologyContributorPlugin providing %d code_structure relationship types",
            len(RPG_RELATIONSHIP_TYPES),
        )
        return RPG_RELATIONSHIP_TYPES

    def get_subtypes(self) -> dict[str, set[str]]:
        """Return the 9 RPG memory subtypes scoped to the semantic memory type."""
        logger.debug(
            "RpgOntologyContributorPlugin providing %d rpg subtypes",
            sum(len(v) for v in RPG_SUBTYPES.values()),
        )
        return RPG_SUBTYPES
