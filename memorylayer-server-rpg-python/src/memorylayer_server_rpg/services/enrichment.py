# SPDX-License-Identifier: Apache-2.0
"""RPG LLM enrichment service — semantic enrichment of code structure graphs.

Ports the TypeScript enrichment logic from scitrera-forge:
  - Feature identification (enrichGraphWithFeatures)
  - Description enrichment (enrichNodeDescriptions)
  - Data flow analysis (analyzeDataFlows)

Requires an LLMService to be configured. Operations gracefully no-op if LLM
returns unparseable responses.
"""

import hashlib
import json
import logging
import re

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.llm import LLMMessage, LLMRequest, LLMRole
from memorylayer_server.services.storage.base import StorageBackend

from .service import RpgService

logger = logging.getLogger(__name__)

# Node type constants used in filtering
_RPG_FILE = "rpg_file"
_RPG_CLASS = "rpg_class"
_RPG_INTERFACE = "rpg_interface"
_RPG_FUNCTION = "rpg_function"
_RPG_COMPONENT = "rpg_component"

# Enrichment configuration
_DESCRIPTION_BATCH_SIZE = 20
_DATA_FLOW_BATCH_SIZE = 5
_MAX_CODEBASE_SUMMARY_LINES = 200
_MAX_IMPORT_ONLY_PAIRS = 20

# Default phases applied when none are specified
DEFAULT_PHASES = ["features", "descriptions", "data_flows"]


def _parse_json_response(text: str):
    """Parse JSON from LLM response, stripping markdown code fences.

    Args:
        text: Raw LLM response text, possibly wrapped in ```json...``` fences.

    Returns:
        Parsed Python object.

    Raises:
        json.JSONDecodeError: If the text cannot be parsed as JSON after cleaning.
    """
    cleaned = text.strip()
    # Strip leading code fence (```json or ```)
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    # Strip trailing code fence
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return json.loads(cleaned)


class RpgEnrichmentService:
    """LLM-based enrichment for RPG code structure graphs.

    Adds a semantic layer on top of structural RPG graphs by using an LLM to:
    - Identify logical feature categories from the code structure
    - Generate improved descriptions for classes and interfaces
    - Detect data flow relationships between components
    """

    def __init__(self, storage: StorageBackend, llm_service):
        self._storage = storage
        self._llm = llm_service
        self._rpg = RpgService(storage)

    async def enrich(
        self,
        workspace_id: str,
        context_id: str = "rpg",
        phases: list[str] | None = None,
    ) -> dict:
        """Run enrichment phases on an RPG graph.

        Args:
            workspace_id: Target workspace.
            context_id: Context partition (default "rpg" for canonical graph).
            phases: Which phases to run. Defaults to all three:
                ["features", "descriptions", "data_flows"].

        Returns:
            Dict with combined stats from all phases run.
        """
        active_phases = phases if phases is not None else DEFAULT_PHASES

        logger.info(
            "Starting RPG enrichment for workspace %s, context %s, phases=%s",
            workspace_id,
            context_id,
            active_phases,
        )

        # Load graph data once — reuse across phases
        nodes = await self._rpg._get_rpg_nodes(workspace_id, context_id=context_id)
        node_id_to_memory_id = {(m.metadata or {}).get("rpg_node_id", m.id): m.id for m in nodes}
        edges = await self._rpg._get_rpg_edges(workspace_id, node_id_to_memory_id)

        combined_stats: dict = {"workspace_id": workspace_id, "context_id": context_id, "phases": {}}

        if "features" in active_phases:
            try:
                stats = await self._identify_features(workspace_id, context_id, nodes)
                combined_stats["phases"]["features"] = stats
            except Exception as exc:
                logger.error("Feature identification failed for workspace %s: %s", workspace_id, exc)
                combined_stats["phases"]["features"] = {"error": str(exc)}

        if "descriptions" in active_phases:
            try:
                stats = await self._enrich_descriptions(workspace_id, context_id, nodes)
                combined_stats["phases"]["descriptions"] = stats
            except Exception as exc:
                logger.error("Description enrichment failed for workspace %s: %s", workspace_id, exc)
                combined_stats["phases"]["descriptions"] = {"error": str(exc)}

        if "data_flows" in active_phases:
            try:
                stats = await self._analyze_data_flows(workspace_id, context_id, nodes, edges)
                combined_stats["phases"]["data_flows"] = stats
            except Exception as exc:
                logger.error("Data flow analysis failed for workspace %s: %s", workspace_id, exc)
                combined_stats["phases"]["data_flows"] = {"error": str(exc)}

        logger.info(
            "RPG enrichment complete for workspace %s: %s",
            workspace_id,
            combined_stats["phases"],
        )

        return combined_stats

    # ------------------------------------------------------------------
    # Phase 1: Feature identification
    # ------------------------------------------------------------------

    async def _identify_features(
        self,
        workspace_id: str,
        context_id: str,
        nodes: list,
    ) -> dict:
        """Port of llm-enricher.ts:enrichGraphWithFeatures().

        Identifies logical feature components from the codebase structure and
        creates rpg_component nodes with contains edges to member files/classes.

        Returns:
            Dict with features_created and edges_created counts.
        """
        # Build codebase summary text and lookups
        codebase_text, path_to_node_id, class_name_to_node_id = self._build_codebase_summary(nodes)

        if not codebase_text.strip():
            logger.info("No file nodes in workspace %s, skipping feature identification", workspace_id)
            return {"features_created": 0, "edges_created": 0}

        # Collect existing node IDs to avoid duplicates
        existing_node_ids = {(m.metadata or {}).get("rpg_node_id", m.id) for m in nodes}

        # Prompt LLM to identify features
        system_prompt = (
            "You are a software architect analyzing a codebase.\n"
            "Your task is to identify logical feature categories that group the code by purpose.\n"
            "Respond ONLY with valid JSON — no prose, no markdown fences."
        )
        user_prompt = (
            "Here is a summary of the codebase files and their symbols:\n\n"
            + codebase_text
            + "\n\nIdentify 5 to 15 logical feature components that group this code by purpose.\n"
            "For each feature, list the file paths (or class names) that primarily implement it.\n"
            "Respond with a JSON array of objects with this exact shape:\n"
            "[\n"
            "  {\n"
            '    "name": "Feature Name",\n'
            '    "description": "One sentence describing what this feature does.",\n'
            '    "members": ["path/to/file.ts", "path/to/other.ts"]\n'
            "  }\n"
            "]"
        )

        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_prompt),
            ],
            max_tokens=4096,
            temperature=0.3,
        )

        try:
            response = await self._llm.complete(request, profile="rpg_enrichment")
            features = _parse_json_response(response.content)
        except Exception as exc:
            logger.warning("Could not parse feature list from LLM response for workspace %s: %s", workspace_id, exc)
            return {"features_created": 0, "edges_created": 0}

        if not isinstance(features, list):
            logger.warning("Feature LLM response is not a list for workspace %s", workspace_id)
            return {"features_created": 0, "edges_created": 0}

        logger.info("LLM identified %d feature components for workspace %s", len(features), workspace_id)

        features_created = 0
        edges_created = 0

        for feature in features:
            if not isinstance(feature, dict):
                continue
            name = feature.get("name", "")
            description = feature.get("description", "")
            members = feature.get("members", [])

            if not name:
                continue

            component_id = "feature:{}".format(name.lower().replace(" ", "-"))

            # Skip if component already exists
            if component_id in existing_node_ids:
                logger.debug("Feature component %s already exists in workspace %s, skipping", component_id, workspace_id)
                continue

            # Create rpg_component node via sync
            component_nodes = [
                {
                    "node_id": component_id,
                    "node_type": _RPG_COMPONENT,
                    "name": name,
                    "description": description,
                    "path": "",
                    "metadata": {"source": "llm-enricher"},
                }
            ]

            # Build contains edges to resolved member nodes
            component_edges = []
            for member in members:
                target_id = (
                    path_to_node_id.get(member) or path_to_node_id.get(member.replace("\\", "/")) or class_name_to_node_id.get(member)
                )
                if not target_id:
                    # Try prefix/substring matching for partial paths
                    for path, node_id in path_to_node_id.items():
                        if path in member or member in path:
                            target_id = node_id
                            break

                if target_id:
                    component_edges.append(
                        {
                            "source_id": component_id,
                            "target_id": target_id,
                            "relationship": "contains",
                            "metadata": {"source": "llm-enricher"},
                        }
                    )

            await self._rpg.sync(
                workspace_id=workspace_id,
                nodes=component_nodes,
                edges=component_edges,
                full_sync=False,
                context_id=context_id,
            )

            existing_node_ids.add(component_id)
            features_created += 1
            edges_created += len(component_edges)

            logger.info(
                "Created feature component '%s' with %d member edges in workspace %s",
                name,
                len(component_edges),
                workspace_id,
            )

        return {"features_created": features_created, "edges_created": edges_created}

    # ------------------------------------------------------------------
    # Phase 2: Description enrichment
    # ------------------------------------------------------------------

    async def _enrich_descriptions(
        self,
        workspace_id: str,
        context_id: str,
        nodes: list,
    ) -> dict:
        """Port of llm-enricher.ts:enrichNodeDescriptions().

        Generates one-line descriptions for class and interface nodes in batches.

        Returns:
            Dict with nodes_enriched count.
        """
        # Collect rpg_class and rpg_interface nodes
        targets = [m for m in nodes if m.subtype in (_RPG_CLASS, _RPG_INTERFACE)]

        if not targets:
            logger.info("No class/interface nodes in workspace %s, skipping description enrichment", workspace_id)
            return {"nodes_enriched": 0}

        logger.info(
            "Enriching descriptions for %d class/interface nodes in workspace %s (batch size %d)",
            len(targets),
            workspace_id,
            _DESCRIPTION_BATCH_SIZE,
        )

        nodes_enriched = 0

        for batch_start in range(0, len(targets), _DESCRIPTION_BATCH_SIZE):
            batch = targets[batch_start : batch_start + _DESCRIPTION_BATCH_SIZE]
            batch_num = batch_start // _DESCRIPTION_BATCH_SIZE + 1
            total_batches = (len(targets) + _DESCRIPTION_BATCH_SIZE - 1) // _DESCRIPTION_BATCH_SIZE

            logger.debug(
                "Processing description batch %d/%d for workspace %s",
                batch_num,
                total_batches,
                workspace_id,
            )

            description_map = await self._generate_descriptions_batch(batch)

            # Apply descriptions by updating memory content in storage
            for mem in batch:
                rpg_node_id = (mem.metadata or {}).get("rpg_node_id", mem.id)
                new_desc = description_map.get(rpg_node_id)
                if not new_desc:
                    continue

                try:
                    content_hash = hashlib.sha256(new_desc.encode()).hexdigest()
                    await self._storage.update_memory(
                        workspace_id=workspace_id,
                        memory_id=mem.id,
                        content=new_desc,
                        content_hash=content_hash,
                    )
                    nodes_enriched += 1
                    logger.debug("Updated description for node %s in workspace %s", rpg_node_id, workspace_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to update description for node %s in workspace %s: %s",
                        rpg_node_id,
                        workspace_id,
                        exc,
                    )

        logger.info("Updated descriptions for %d nodes in workspace %s", nodes_enriched, workspace_id)
        return {"nodes_enriched": nodes_enriched}

    async def _generate_descriptions_batch(self, batch: list) -> dict:
        """Generate one-line descriptions for a batch of class/interface nodes.

        Args:
            batch: List of Memory objects (rpg_class or rpg_interface).

        Returns:
            Dict mapping rpg_node_id -> description string.
        """
        result: dict = {}
        if not batch:
            return result

        item_lines = []
        for i, mem in enumerate(batch):
            meta = mem.metadata or {}
            node_type = "class" if mem.subtype == _RPG_CLASS else "interface"
            name = meta.get("rpg_name", "")
            path = meta.get("rpg_path", "")
            item_lines.append(f'{i}: {node_type} "{name}" in {path}')

        item_list = "\n".join(item_lines)

        system_prompt = (
            "You are a software documentation assistant.\n"
            "Generate concise one-line descriptions for code symbols.\n"
            "Respond ONLY with valid JSON — no prose, no markdown fences."
        )
        user_prompt = (
            "Generate a one-line description for each of the following code symbols.\n"
            "Use the symbol name and file path to infer purpose.\n\n"
            + item_list
            + "\n\nRespond with a JSON object mapping each index (as a string) to its description:\n"
            "{\n"
            '  "0": "Manages user authentication sessions.",\n'
            '  "1": "Represents a single database record entity."\n'
            "}"
        )

        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_prompt),
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        try:
            response = await self._llm.complete(request, profile="rpg_enrichment")
            parsed = _parse_json_response(response.content)
        except Exception as exc:
            logger.warning("Could not parse descriptions from LLM response: %s", exc)
            return result

        if not isinstance(parsed, dict):
            logger.warning("Description LLM response is not a dict")
            return result

        for idx_str, desc in parsed.items():
            try:
                i = int(idx_str)
            except (ValueError, TypeError):
                continue
            if i < 0 or i >= len(batch):
                continue
            mem = batch[i]
            if not isinstance(desc, str):
                continue
            rpg_node_id = (mem.metadata or {}).get("rpg_node_id", mem.id)
            result[rpg_node_id] = desc

        return result

    # ------------------------------------------------------------------
    # Phase 3: Data flow analysis
    # ------------------------------------------------------------------

    async def _analyze_data_flows(
        self,
        workspace_id: str,
        context_id: str,
        nodes: list,
        edges: list,
    ) -> dict:
        """Port of data-flow-analyzer.ts:analyzeDataFlows().

        Finds cross-component pairs connected by import/invocation edges and
        uses the LLM to identify data flows between them.

        Returns:
            Dict with flows_created and pairs_analyzed counts.
        """
        # Build node map for fast lookup: memory_id -> Memory
        node_by_memory_id = {m.id: m for m in nodes}

        # Build rpg_node_id -> Memory map
        node_by_rpg_id: dict[str, object] = {}
        for m in nodes:
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            node_by_rpg_id[rpg_id] = m

        # Find cross-component pairs: files connected by imports and/or invocations
        pairs = self._find_cross_component_pairs(nodes, edges, node_by_memory_id)

        if not pairs:
            logger.info("No cross-component pairs found in workspace %s, skipping data flow analysis", workspace_id)
            return {"flows_created": 0, "pairs_analyzed": 0}

        logger.info("Analyzing data flows for %d component pairs in workspace %s", len(pairs), workspace_id)

        flows_created = 0
        pairs_analyzed = 0

        for batch_start in range(0, len(pairs), _DATA_FLOW_BATCH_SIZE):
            batch = pairs[batch_start : batch_start + _DATA_FLOW_BATCH_SIZE]

            try:
                new_flows = await self._analyze_data_flow_batch(
                    workspace_id,
                    batch,
                    nodes,
                    node_by_rpg_id,
                )
                for flow in new_flows:
                    # Resolve node IDs to memory IDs for association creation
                    src_mem = node_by_rpg_id.get(flow["source_id"])
                    tgt_mem = node_by_rpg_id.get(flow["target_id"])
                    if not src_mem or not tgt_mem:
                        logger.debug(
                            "Skipping data flow %s->%s: node not found in workspace %s",
                            flow["source_id"],
                            flow["target_id"],
                            workspace_id,
                        )
                        continue

                    assoc_input = AssociateInput(
                        source_id=src_mem.id,
                        target_id=tgt_mem.id,
                        relationship="data_flow",
                        strength=1.0,
                        metadata={
                            "data_id": flow["data_id"],
                            "data_type": flow["data_type"],
                            "transformation": flow["transformation"],
                            "rpg_source_node_id": flow["source_id"],
                            "rpg_target_node_id": flow["target_id"],
                            "source": "llm-data-flow-analyzer",
                        },
                    )
                    try:
                        await self._storage.create_association(
                            workspace_id=workspace_id,
                            input=assoc_input,
                        )
                        flows_created += 1
                    except Exception as exc:
                        logger.warning(
                            "Failed to create data_flow association in workspace %s: %s",
                            workspace_id,
                            exc,
                        )

                pairs_analyzed += len(batch)
            except Exception as exc:
                logger.error(
                    "Data flow batch %d failed for workspace %s: %s",
                    batch_start // _DATA_FLOW_BATCH_SIZE + 1,
                    workspace_id,
                    exc,
                )

        logger.info(
            "Data flow analysis complete for workspace %s: %d flows created from %d pairs",
            workspace_id,
            flows_created,
            pairs_analyzed,
        )

        return {"flows_created": flows_created, "pairs_analyzed": pairs_analyzed}

    def _find_cross_component_pairs(
        self,
        nodes: list,
        edges: list,
        node_by_memory_id: dict,
    ) -> list[dict]:
        """Find pairs of file nodes connected by import and/or invocation edges.

        Port of data-flow-analyzer.ts:findCrossComponentPairs().

        Returns list of dicts with source_id and target_id (rpg_node_ids).
        """
        # Build lookup: memory_id -> rpg_node_id
        mem_to_rpg: dict[str, str] = {}
        for m in nodes:
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            mem_to_rpg[m.id] = rpg_id

        # Build lookup: rpg_node_id -> subtype for file detection
        rpg_subtype: dict[str, str] = {}
        for m in nodes:
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            rpg_subtype[rpg_id] = m.subtype or ""

        def _get_file_for_node(rpg_id: str) -> str | None:
            """Get file-level rpg_node_id for a given rpg_node_id."""
            # Symbol nodes have format "path/to/file.ext:ClassName"
            colon_idx = rpg_id.find(":")
            if colon_idx >= 0:
                return rpg_id[:colon_idx]
            # Could already be a file node
            if rpg_subtype.get(rpg_id) == _RPG_FILE:
                return rpg_id
            return None

        import_pairs: set[str] = set()
        invoke_pairs: set[str] = set()

        for assoc in edges:
            src_rpg = mem_to_rpg.get(assoc.source_id, "")
            tgt_rpg = mem_to_rpg.get(assoc.target_id, "")
            rel = assoc.relationship

            if rel == "imports":
                if src_rpg and tgt_rpg:
                    import_pairs.add(f"{src_rpg}|{tgt_rpg}")
            elif rel == "invokes":
                src_file = _get_file_for_node(src_rpg)
                tgt_file = _get_file_for_node(tgt_rpg)
                if src_file and tgt_file and src_file != tgt_file:
                    invoke_pairs.add(f"{src_file}|{tgt_file}")

        pairs: list[dict] = []

        # Pairs with both import AND invocation — strongest data flow candidates
        for pair_key in import_pairs:
            if pair_key in invoke_pairs:
                src, tgt = pair_key.split("|", 1)
                pairs.append({"source_id": src, "target_id": tgt})

        # Pure import pairs (limited to top-N for cost control)
        for pair_key in import_pairs:
            if pair_key not in invoke_pairs and len(pairs) < _MAX_IMPORT_ONLY_PAIRS:
                src, tgt = pair_key.split("|", 1)
                pairs.append({"source_id": src, "target_id": tgt})

        return pairs

    async def _analyze_data_flow_batch(
        self,
        workspace_id: str,
        pairs: list[dict],
        nodes: list,
        node_by_rpg_id: dict,
    ) -> list[dict]:
        """Analyze a batch of component pairs for data flows via LLM.

        Port of data-flow-analyzer.ts:analyzeDataFlowBatch().

        Returns list of resolved flow dicts with source_id, target_id, data_id,
        data_type, and transformation keys.
        """
        # Build path-to-node-id lookup for resolving LLM responses
        path_to_node_id: dict[str, str] = {}
        for pair in pairs:
            path_to_node_id[pair["source_id"]] = pair["source_id"]
            path_to_node_id[pair["target_id"]] = pair["target_id"]
            # Also map just the filename for fuzzy matching
            src_name = pair["source_id"].rsplit("/", 1)[-1]
            tgt_name = pair["target_id"].rsplit("/", 1)[-1]
            if src_name not in path_to_node_id:
                path_to_node_id[src_name] = pair["source_id"]
            if tgt_name not in path_to_node_id:
                path_to_node_id[tgt_name] = pair["target_id"]

        # Build compact context for the LLM
        pair_descriptions = []
        for pair in pairs:
            src_mem = node_by_rpg_id.get(pair["source_id"])
            tgt_mem = node_by_rpg_id.get(pair["target_id"])
            src_name = (src_mem.metadata or {}).get("rpg_name", pair["source_id"]) if src_mem else pair["source_id"]
            tgt_name = (tgt_mem.metadata or {}).get("rpg_name", pair["target_id"]) if tgt_mem else pair["target_id"]

            # Get symbols defined within each file node
            src_prefix = pair["source_id"] + ":"
            tgt_prefix = pair["target_id"] + ":"
            src_symbols = [
                "{}: {}".format(
                    (m.subtype or "").replace("rpg_", ""),
                    (m.metadata or {}).get("rpg_name", ""),
                )
                for m in nodes
                if ((m.metadata or {}).get("rpg_node_id", "")).startswith(src_prefix)
            ][:10]
            tgt_symbols = [
                "{}: {}".format(
                    (m.subtype or "").replace("rpg_", ""),
                    (m.metadata or {}).get("rpg_name", ""),
                )
                for m in nodes
                if ((m.metadata or {}).get("rpg_node_id", "")).startswith(tgt_prefix)
            ][:10]

            pair_descriptions.append(
                "Pair: {} \u2192 {}\n  Source ({}): {}\n  Target ({}): {}".format(
                    pair["source_id"],
                    pair["target_id"],
                    src_name,
                    ", ".join(src_symbols) or "no symbols",
                    tgt_name,
                    ", ".join(tgt_symbols) or "no symbols",
                )
            )

        system_prompt = (
            "You are a code architecture analyst. Identify data flows between code components.\n"
            "For each component pair, determine what data flows from source to target:\n"
            "- What data is passed (data_id: short identifier)\n"
            '- What type it is (data_type: e.g. "user_input", "config", "model_data", "api_response")\n'
            "- How it's transformed (transformation: brief description)\n\n"
            "Respond in JSON array format:\n"
            '[{"source": "file.py", "target": "other.py", "data_id": "user_data", '
            '"data_type": "model_data", "transformation": "validated and serialized"}]\n\n'
            "Only include flows where there's a clear data dependency. Skip pairs with no meaningful data flow."
        )
        user_prompt = "Analyze data flows between these component pairs:\n\n" + "\n\n".join(pair_descriptions)

        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_prompt),
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        try:
            response = await self._llm.complete(request, profile="rpg_enrichment")
            raw = response.content

            # Extract JSON array from response (LLM may include prose)
            json_match = re.search(r"\[[\s\S]*\]", raw)
            if not json_match:
                return []
            flows_raw = json.loads(json_match.group(0))
        except Exception as exc:
            logger.warning("Could not parse data flow response from LLM: %s", exc)
            return []

        result: list[dict] = []
        for f in flows_raw:
            if not isinstance(f, dict):
                continue

            src_raw = f.get("source", "")
            tgt_raw = f.get("target", "")

            # Resolve LLM-returned paths back to real node IDs
            resolved_source = path_to_node_id.get(src_raw) or path_to_node_id.get(src_raw.rsplit("/", 1)[-1])
            resolved_target = path_to_node_id.get(tgt_raw) or path_to_node_id.get(tgt_raw.rsplit("/", 1)[-1])

            if not resolved_source or not resolved_target:
                logger.debug("Skipping unresolvable data flow %s -> %s", src_raw, tgt_raw)
                continue

            result.append(
                {
                    "source_id": resolved_source,
                    "target_id": resolved_target,
                    "data_id": f.get("data_id", ""),
                    "data_type": f.get("data_type", ""),
                    "transformation": f.get("transformation", ""),
                }
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_codebase_summary(
        self,
        nodes: list,
    ) -> tuple[str, dict, dict]:
        """Build compact textual summary of codebase for the LLM prompt.

        Port of llm-enricher.ts:buildCodebaseSummary().

        Returns:
            Tuple of (summary_text, path_to_node_id, class_name_to_node_id).
        """
        file_nodes = [m for m in nodes if m.subtype == _RPG_FILE]
        class_nodes = [m for m in nodes if m.subtype in (_RPG_CLASS, _RPG_INTERFACE)]
        [m for m in nodes if m.subtype == _RPG_FUNCTION]

        # Build class and function name -> node_id lookups
        class_name_to_node_id: dict[str, str] = {}
        for m in class_nodes:
            name = (m.metadata or {}).get("rpg_name", "")
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            if name:
                class_name_to_node_id[name] = rpg_id

        # Build path -> node_id lookup
        path_to_node_id: dict[str, str] = {}
        for m in file_nodes:
            path = (m.metadata or {}).get("rpg_path", "")
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            if path:
                path_to_node_id[path] = rpg_id

        # Map memory_id -> rpg_node_id for edge lookup
        mem_to_rpg: dict[str, str] = {}
        for m in nodes:
            mem_to_rpg[m.id] = (m.metadata or {}).get("rpg_node_id", m.id)

        # Build parent_id -> children mapping using rpg_parent_id metadata
        children_by_parent: dict[str, list] = {}
        for m in nodes:
            parent_rpg_id = (m.metadata or {}).get("rpg_parent_id")
            if parent_rpg_id:
                children_by_parent.setdefault(parent_rpg_id, []).append(m)

        # Build rpg_id -> Memory lookup
        node_by_rpg_id: dict[str, object] = {}
        for m in nodes:
            rpg_id = (m.metadata or {}).get("rpg_node_id", m.id)
            node_by_rpg_id[rpg_id] = m

        lines: list[str] = []
        for file_mem in file_nodes:
            file_rpg_id = (file_mem.metadata or {}).get("rpg_node_id", file_mem.id)
            file_path = (file_mem.metadata or {}).get("rpg_path", file_rpg_id)

            children = children_by_parent.get(file_rpg_id, [])
            classes: list[str] = []
            functions: list[str] = []

            for child in children:
                child_name = (child.metadata or {}).get("rpg_name", "")
                if child.subtype in (_RPG_CLASS, _RPG_INTERFACE):
                    classes.append(child_name)
                elif child.subtype == _RPG_FUNCTION:
                    functions.append(child_name)

            parts: list[str] = []
            if classes:
                parts.append("classes: {}".format(", ".join(classes[:8])))
            if functions:
                parts.append("functions: {}".format(", ".join(functions[:8])))

            suffix = " [{}]".format("; ".join(parts)) if parts else ""
            lines.append(f"  {file_path}{suffix}")

        # Cap to maximum lines
        truncated = len(lines) > _MAX_CODEBASE_SUMMARY_LINES
        summary_lines = lines[:_MAX_CODEBASE_SUMMARY_LINES]
        if truncated:
            summary_lines.append(f"  ... ({len(lines) - _MAX_CODEBASE_SUMMARY_LINES} more files omitted)")

        text = "\n".join(summary_lines)
        return text, path_to_node_id, class_name_to_node_id
