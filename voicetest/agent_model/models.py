"""
Pydantic models for the Agent Model — the durable artifact at the center of the
product.

Instead of `description -> scenarios` (throwing the description away), the platform
extracts `description -> AgentModel -> scenarios`. The AgentModel is a persisted,
developer-confirmable understanding of the agent: its goals, the information it
must collect, the tools it uses, the business rules it must obey, the risks it
faces, and a coarse conversation graph (nodes + edges).

Everything downstream — grounded scenario generation, graph coverage, root-cause
analysis — references THIS object, not the raw prompt and not the transcript.

These models are deliberately forgiving (`extra='ignore'`) because they are
populated from an LLM extraction that may add or omit fields as it improves. The
extractor coerces missing fields to sensible empties so the UI never crashes on a
partial model.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "node"


class RequiredInfo(BaseModel):
    """One piece of information the agent must collect from the caller."""

    model_config = {"extra": "ignore"}

    key: str  # machine-ish key, e.g. "phone", "date"
    label: str  # human label, e.g. "Callback number"
    required: bool = True


class Tool(BaseModel):
    """An external capability the agent invokes (calendar, CRM, payments, …)."""

    model_config = {"extra": "ignore"}

    name: str
    purpose: str = ""


class BusinessRule(BaseModel):
    """A policy the agent must obey (e.g. 'never double-book')."""

    model_config = {"extra": "ignore"}

    rule: str


class Risk(BaseModel):
    """A failure mode worth testing (e.g. 'calendar API timeout')."""

    model_config = {"extra": "ignore"}

    description: str
    # Optional pointer to the node this risk attaches to, if the extractor knows.
    node_id: str = ""


class GraphNode(BaseModel):
    """One step in the coarse conversation graph."""

    model_config = {"extra": "ignore"}

    id: str
    label: str
    objective: str = ""


class GraphEdge(BaseModel):
    """A transition between two nodes; `condition` labels the branch."""

    model_config = {"extra": "ignore"}

    source: str  # node id
    target: str  # node id
    condition: str = ""  # e.g. "success", "failure", "no availability"


class AgentModel(BaseModel):
    """The platform's understanding of one voice agent. Persisted + editable."""

    model_config = {"extra": "ignore"}

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # The raw text this model was extracted from — kept so re-extraction/editing
    # can reference the source of truth.
    source_description: str = ""
    # Whether a developer has reviewed/confirmed the extracted model.
    confirmed: bool = False

    goal: str = ""
    required_info: list[RequiredInfo] = []
    tools: list[Tool] = []
    business_rules: list[BusinessRule] = []
    risks: list[Risk] = []
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    def ensure_node_ids(self) -> AgentModel:
        """
        Guarantee every node has a stable id and every edge references real nodes.

        The extractor may emit human labels without ids, or edges that name nodes
        by label. This normalizes so the graph is always renderable.
        """
        # Assign ids to any node missing one (derive from label, dedupe).
        seen: set[str] = set()
        label_to_id: dict[str, str] = {}
        for node in self.nodes:
            nid = node.id or _slug(node.label)
            base = nid
            i = 2
            while nid in seen:
                nid = f"{base}-{i}"
                i += 1
            node.id = nid
            seen.add(nid)
            if node.label:
                label_to_id.setdefault(node.label, nid)
            label_to_id.setdefault(nid, nid)

        # Rewrite edge endpoints that reference labels instead of ids, and drop
        # edges pointing at nodes that don't exist.
        valid: list[GraphEdge] = []
        for edge in self.edges:
            src = label_to_id.get(edge.source, edge.source)
            tgt = label_to_id.get(edge.target, edge.target)
            if src in seen and tgt in seen:
                edge.source, edge.target = src, tgt
                valid.append(edge)
        self.edges = valid
        return self


def coerce_agent_model(data: dict[str, Any], source_description: str) -> AgentModel:
    """
    Build an AgentModel from a loosely-shaped extraction dict.

    Accepts several shapes the LLM might emit (e.g. required_info as list[str] or
    list[dict]) and normalizes them, so a slightly-off extraction still yields a
    renderable model rather than a validation error.
    """
    model = AgentModel(source_description=source_description)
    model.goal = str(data.get("goal", "") or "")

    # required_info — accept ["name", "phone"] or [{"key","label","required"}]
    for item in data.get("required_info", []) or []:
        if isinstance(item, str):
            model.required_info.append(RequiredInfo(key=_slug(item), label=item))
        elif isinstance(item, dict):
            label = item.get("label") or item.get("key") or ""
            model.required_info.append(
                RequiredInfo(
                    key=item.get("key") or _slug(label),
                    label=label or item.get("key", ""),
                    required=bool(item.get("required", True)),
                )
            )

    # tools — accept ["Calendar API"] or [{"name","purpose"}]
    for item in data.get("tools", []) or []:
        if isinstance(item, str):
            model.tools.append(Tool(name=item))
        elif isinstance(item, dict):
            model.tools.append(
                Tool(name=item.get("name", ""), purpose=item.get("purpose", ""))
            )

    # business_rules — accept ["never double-book"] or [{"rule"}]
    for item in data.get("business_rules", []) or []:
        if isinstance(item, str):
            model.business_rules.append(BusinessRule(rule=item))
        elif isinstance(item, dict):
            model.business_rules.append(BusinessRule(rule=item.get("rule", "")))

    # risks — accept ["calendar timeout"] or [{"description","node_id"}]
    for item in data.get("risks", []) or []:
        if isinstance(item, str):
            model.risks.append(Risk(description=item))
        elif isinstance(item, dict):
            model.risks.append(
                Risk(
                    description=item.get("description", ""),
                    node_id=item.get("node_id", ""),
                )
            )

    # nodes
    for item in data.get("nodes", []) or []:
        if isinstance(item, str):
            model.nodes.append(GraphNode(id=_slug(item), label=item))
        elif isinstance(item, dict):
            label = item.get("label") or item.get("id") or ""
            model.nodes.append(
                GraphNode(
                    id=item.get("id") or _slug(label),
                    label=label or item.get("id", ""),
                    objective=item.get("objective", ""),
                )
            )

    # edges
    for item in data.get("edges", []) or []:
        if isinstance(item, dict):
            model.edges.append(
                GraphEdge(
                    source=item.get("source", ""),
                    target=item.get("target", ""),
                    condition=item.get("condition", ""),
                )
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            model.edges.append(GraphEdge(source=str(item[0]), target=str(item[1])))

    return model.ensure_node_ids()
