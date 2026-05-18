"""
Ontology Contributor Plugin Base.

Base class for plugins that contribute relationship types to the ontology
service at startup. Plugins are auto-discovered via the
``EXT_MULTI_ONTOLOGY_CONTRIBUTORS`` multi-extension point and merged into
the live ontology service during its ``async_ready`` hook.
"""

from abc import abstractmethod

from scitrera_app_framework import Plugin, Variables

from .._constants import EXT_MULTI_ONTOLOGY_CONTRIBUTORS


class OntologyContributorPlugin(Plugin):
    """Plugins that contribute ontology elements at startup.

    Subclasses override ``get_relationship_types()`` to return a mapping of
    relationship type name to metadata dict (description, symmetric,
    transitive, inverse, category). Discovered via the standard
    multi-extension scan; merged into the live ontology service during its
    ``async_ready`` hook.

    Use this (pull) path for static contributions. For dynamic or runtime
    contributions, prefer calling ``OntologyService.extend_ontology(...)``
    directly (push path).
    """

    def name(self) -> str:
        """Human-readable contributor name (used for diagnostics/source tag)."""
        return self.__class__.__name__

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_ONTOLOGY_CONTRIBUTORS

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def is_enabled(self, v: Variables) -> bool:
        """Disable 'single' extension for multi-extension plugins."""
        return False

    def initialize(self, v: Variables, logger):
        # Stateless wrapper; the ontology service queries the plugin instance
        # later via async_ready.
        return self

    @abstractmethod
    def get_relationship_types(self) -> dict[str, dict]:
        """Return relationship types contributed by this plugin.

        Returns:
            Mapping of ``type_name -> {description, symmetric, transitive,
            inverse, category}``.
        """
        ...

    def get_subtypes(self) -> dict[str, set[str]]:
        """Return memory subtypes contributed by this plugin.

        Subclasses may override to contribute domain-specific memory
        subtypes that aren't part of the OSS-known subtype set. Default
        is an empty mapping (contribute nothing).

        Returns:
            Mapping of ``memory_type -> set of subtype strings``. Use
            ``"*"`` as the memory_type key to contribute subtypes that
            apply to any memory type.
        """
        return {}
