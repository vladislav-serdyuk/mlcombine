"""ModelBuilder — resolves the model DAG from ``ModelNode`` definitions."""

from __future__ import annotations

import logging
from typing import Any

from mlcombine.core.schemas.config import ModelNode
from mlcombine.core.schemas.blueprint import ModelBlueprint


logger = logging.getLogger(__name__)


class ModelBuilder:
    """Resolves a list of ``ModelNode`` entries into a ``ModelBlueprint`` tree.

    Nodes can reference one another via ``model`` (single reference) or
    ``models`` (list of references).  References are resolved to
    ``ModelBlueprint`` objects, creating a lazy tree that can be
    materialised later via ``blueprint.build()``.
    """

    def __init__(self) -> None:
        self._built_dict: dict[str, ModelBlueprint] = {}

    @property
    def built_dict(self) -> dict[str, ModelBlueprint]:
        """Return all built blueprints keyed by node id."""
        return dict(self._built_dict)

    def build_all(
        self,
        nodes: list[ModelNode],
        *,
        task_type: str = "regression",
        num_classes: int | None = None,
        input_size: int | None = None,
    ) -> ModelBlueprint:
        """Build all nodes in dependency order and return the final blueprint.

        Args:
            nodes: ModelNode definitions from the pipeline config.
            task_type: ``TaskType`` value (string).
            num_classes: Number of classes for classification.
            input_size: Number of features (auto-detected from training data).

        Returns:
            A ``ModelBlueprint`` for the last node (fully resolved with
            blueprint dependencies).
        """
        self._built_dict = {}
        built: dict[str, ModelBlueprint] = self._built_dict
        ordered = self._topological_sort(nodes)
        logger.info("Building %d model nodes (task_type=%s, num_classes=%s, input_size=%s)", len(nodes), task_type, num_classes, input_size)

        for node in ordered:
            deps = self._resolve_deps(node, built)

            blueprint = ModelBlueprint(
                provider=node.provider,
                params=dict(node.params),
                model=deps.get("model"),
                models=deps.get("models"),
                task_type=task_type,
                num_classes=num_classes,
                input_size=input_size,
            )

            node_id = node.id or node.provider
            built[node_id] = blueprint
            logger.debug("Built node %s (provider=%s)", node_id, node.provider)

        logger.info("Final model node: %s (provider=%s)", ordered[-1].id or ordered[-1].provider, ordered[-1].provider)
        return built[ordered[-1].id or ordered[-1].provider]

    def _resolve_deps(
        self,
        node: ModelNode,
        built: dict[str, ModelBlueprint],
    ) -> dict[str, Any]:
        """Resolve ``model`` / ``models`` references from already-built blueprints."""
        deps: dict[str, Any] = {}
        if node.model:
            ref = built.get(node.model)
            if ref is None:
                raise ValueError(f"Node {node.provider!r} references unknown model={node.model!r}. Available: {list(built)}")
            deps["model"] = ref
        if node.models:
            resolved: list[ModelBlueprint] = []
            for ref_name in node.models:
                ref = built.get(ref_name)
                if ref is None:
                    raise ValueError(f"Node {node.provider!r} references unknown model={ref_name!r}. Available: {list(built)}")
                resolved.append(ref)
            deps["models"] = resolved
        return deps

    @staticmethod
    def _topological_sort(nodes: list[ModelNode]) -> list[ModelNode]:
        """Topological sort of nodes based on ``model``/``models`` references.

        Returns nodes in an order where dependencies are built before dependents.
        """
        id_map: dict[str, ModelNode] = {}
        node_list: list[ModelNode] = []
        for node in nodes:
            nid = node.id or node.provider
            id_map[nid] = node
            node_list.append(node)

        visited: set[str] = set()
        result: list[ModelNode] = []

        def _visit(n: ModelNode) -> None:
            nid = n.id or n.provider
            if nid in visited:
                return
            visited.add(nid)
            deps: list[str] = []
            if n.model:
                deps.append(n.model)
            if n.models:
                deps.extend(n.models)
            for dep_id in deps:
                dep_node = id_map.get(dep_id)
                if dep_node:
                    _visit(dep_node)
            result.append(n)

        for n in node_list:
            _visit(n)

        return result
