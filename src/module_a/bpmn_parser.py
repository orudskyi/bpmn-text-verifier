"""Module A — BPMN 2.0 XML parser.

Parses a BPMN 2.0 XML file into a :class:`~src.models.BPMNGraph` Pydantic model.
Only structural/semantic information is extracted; all diagram-visual data
(``bpmndi`` namespace) is intentionally ignored.

Typical usage::

    from src.module_a.bpmn_parser import parse_bpmn

    graph = parse_bpmn("data/dispatch/Dispatch-of-goods.bpmn")
    print(graph.process_name)       # "Dispatch of goods Computer Hardware Shop"
    print(len(graph.nodes))         # 14 (tasks + gateways + events)
"""

import logging
import re
from pathlib import Path

from lxml import etree

from src.models import BPMNEdge, BPMNGraph, BPMNNode, BPMNNodeType

logger = logging.getLogger(__name__)

# ── BPMN XML namespace ────────────────────────────────────────────────────────
_BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
_NSMAP = {"bpmn": _BPMN_NS}

# ── Mapping: localname → BPMNNodeType ────────────────────────────────────────
_TAG_TO_NODE_TYPE: dict[str, BPMNNodeType] = {
    "task": BPMNNodeType.TASK,
    "startEvent": BPMNNodeType.START_EVENT,
    "endEvent": BPMNNodeType.END_EVENT,
    "intermediateCatchEvent": BPMNNodeType.INTERMEDIATE_EVENT,
    "intermediateThrowEvent": BPMNNodeType.INTERMEDIATE_EVENT,
    "exclusiveGateway": BPMNNodeType.EXCLUSIVE_GATEWAY,
    "inclusiveGateway": BPMNNodeType.INCLUSIVE_GATEWAY,
    "parallelGateway": BPMNNodeType.PARALLEL_GATEWAY,
    "eventBasedGateway": BPMNNodeType.EVENT_BASED_GATEWAY,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clean_name(raw: str | None) -> str:
    """Normalise a BPMN element name.

    Replaces XML newline entities (``&#10;`` decoded as ``\\n``) and any other
    whitespace sequences with a single space, then strips leading/trailing
    whitespace.

    Args:
        raw: Raw name string as decoded by lxml (may contain ``\\n``).

    Returns:
        Cleaned, single-line name string; empty string if *raw* is ``None``.
    """
    if not raw:
        return ""
    return re.sub(r"\s+", " ", raw).strip()


def _local_name(element: etree._Element) -> str:
    """Return the XML local name (tag without namespace URI).

    Args:
        element: An lxml element such as ``{http://…}task``.

    Returns:
        Local part of the tag, e.g. ``"task"``.
    """
    return etree.QName(element.tag).localname


# ── Public API ────────────────────────────────────────────────────────────────


def parse_bpmn(file_path: str) -> BPMNGraph:
    """Parse a BPMN 2.0 XML file into a structured graph representation.

    The function extracts:

    * **Nodes** — all flow elements (tasks, events, gateways) with their type,
      name, and swim-lane assignment.
    * **Edges** — all sequence flows with source/target references and optional
      condition labels.
    * **Lanes** — mapping of swim-lane name → list of contained node IDs.
    * **Process name** — taken from the ``<bpmn:participant>`` element inside
      ``<bpmn:collaboration>``; falls back to the process ``id`` attribute if no
      collaboration is present.

    All diagram-visual information (``bpmndi`` namespace) is ignored.

    Args:
        file_path: Path to the ``.bpmn`` XML file.

    Returns:
        :class:`~src.models.BPMNGraph` with all process nodes, edges, and lane
        assignments.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file cannot be parsed as valid BPMN XML or does not
            contain a ``<bpmn:process>`` element.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"BPMN file not found: {file_path}")

    logger.info("Parsing BPMN file: %s", path)

    try:
        tree = etree.parse(str(path))  # noqa: S320  (trusted local files)
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"Invalid XML in BPMN file '{file_path}': {exc}") from exc

    root = tree.getroot()

    # ── 1. Process name from <bpmn:collaboration> ────────────────────────────
    process_name = _extract_process_name(root)
    logger.debug("Process name: %r", process_name)

    # ── 2. Locate the <bpmn:process> element ─────────────────────────────────
    process_elements = root.findall("bpmn:process", _NSMAP)
    if not process_elements:
        raise ValueError(f"No <bpmn:process> element found in '{file_path}'.")
    # Use first process (single-process diagrams are the common case)
    process_el = process_elements[0]

    # ── 3. Extract lane map: lane_name → {node_id, …} ────────────────────────
    lanes: dict[str, list[str]] = _extract_lanes(process_el)
    # Invert for quick node→lane lookup
    node_to_lane: dict[str, str] = {
        node_id: lane_name
        for lane_name, node_ids in lanes.items()
        for node_id in node_ids
    }

    # ── 4. Extract flow nodes ─────────────────────────────────────────────────
    nodes: list[BPMNNode] = _extract_nodes(process_el, node_to_lane)
    logger.debug("Extracted %d nodes", len(nodes))

    # ── 5. Extract sequence flows ─────────────────────────────────────────────
    edges: list[BPMNEdge] = _extract_edges(process_el)
    logger.debug("Extracted %d edges", len(edges))

    # ── 6. Validate edge references ───────────────────────────────────────────
    node_ids = {n.id for n in nodes}
    invalid_edges = [
        e for e in edges if e.source not in node_ids or e.target not in node_ids
    ]
    if invalid_edges:
        logger.warning(
            "Found %d edge(s) referencing unknown node IDs: %s",
            len(invalid_edges),
            [e.id for e in invalid_edges],
        )

    graph = BPMNGraph(
        process_name=process_name,
        nodes=nodes,
        edges=edges,
        lanes=lanes,
    )
    logger.info(
        "Parsed %s: %d nodes, %d edges, %d lanes",
        process_name,
        len(nodes),
        len(edges),
        len(lanes),
    )
    return graph


# ── Private helpers ───────────────────────────────────────────────────────────


def _extract_process_name(root: etree._Element) -> str:
    """Extract the human-readable process name from a collaboration participant.

    Falls back to the process element's ``id`` attribute, then to ``"Unknown"``.

    Args:
        root: The root ``<bpmn:definitions>`` element.

    Returns:
        Cleaned process name string.
    """
    # Try collaboration → participant name (most descriptive)
    participant = root.find("bpmn:collaboration/bpmn:participant", _NSMAP)
    if participant is not None:
        raw = participant.get("name", "")
        name = _clean_name(raw)
        if name:
            return name

    # Fallback: process id
    process_el = root.find("bpmn:process", _NSMAP)
    if process_el is not None:
        process_id = process_el.get("id", "Unknown")
        logger.debug("No participant name found; using process id: %r", process_id)
        return process_id

    return "Unknown"


def _extract_lanes(process_el: etree._Element) -> dict[str, list[str]]:
    """Extract swim-lane definitions from a ``<bpmn:process>`` element.

    Args:
        process_el: The ``<bpmn:process>`` XML element.

    Returns:
        Dictionary mapping lane name → ordered list of flow-node IDs.
        Returns an empty dict if no ``<bpmn:laneSet>`` is present.
    """
    lanes: dict[str, list[str]] = {}
    for lane_el in process_el.findall(".//bpmn:lane", _NSMAP):
        lane_name = _clean_name(lane_el.get("name", ""))
        node_refs = [
            ref_el.text.strip()
            for ref_el in lane_el.findall("bpmn:flowNodeRef", _NSMAP)
            if ref_el.text
        ]
        if lane_name:
            lanes[lane_name] = node_refs
    return lanes


def _extract_nodes(
    process_el: etree._Element,
    node_to_lane: dict[str, str],
) -> list[BPMNNode]:
    """Extract all BPMN flow nodes from a ``<bpmn:process>`` element.

    Only direct children of the process element are inspected (lane elements
    are nested inside ``<bpmn:laneSet>`` and ignored here).

    Args:
        process_el: The ``<bpmn:process>`` XML element.
        node_to_lane: Pre-built mapping of node ID → lane name.

    Returns:
        List of :class:`~src.models.BPMNNode` instances.
    """
    nodes: list[BPMNNode] = []
    for child in process_el:
        local = _local_name(child)
        node_type = _TAG_TO_NODE_TYPE.get(local)
        if node_type is None:
            continue  # skip sequenceFlow, laneSet, etc.

        node_id = child.get("id", "")
        name = _clean_name(child.get("name"))
        lane = node_to_lane.get(node_id, "")

        nodes.append(BPMNNode(id=node_id, type=node_type, name=name, lane=lane))

    return nodes


def _extract_edges(process_el: etree._Element) -> list[BPMNEdge]:
    """Extract all sequence flows from a ``<bpmn:process>`` element.

    Args:
        process_el: The ``<bpmn:process>`` XML element.

    Returns:
        List of :class:`~src.models.BPMNEdge` instances.
    """
    edges: list[BPMNEdge] = []
    for sf in process_el.findall("bpmn:sequenceFlow", _NSMAP):
        edge_id = sf.get("id", "")
        source = sf.get("sourceRef", "")
        target = sf.get("targetRef", "")
        name = _clean_name(sf.get("name"))
        edges.append(BPMNEdge(id=edge_id, source=source, target=target, name=name))
    return edges
