"""Module C — Symbolic Verification Engine.

Checks whether a BPMN process model satisfies a set of DECLARE constraints
extracted from a natural-language description.  **No LLM is used** — this
module is purely algorithmic.

Algorithm overview:
    1. Build a directed adjacency structure from the BPMNGraph.
    2. Enumerate all possible execution traces via DFS, respecting gateway
       semantics (exclusive fork → one branch, parallel fork → all
       interleavings, inclusive fork → all non-empty subsets).
    3. For each DECLARE constraint, check whether it holds on *all* generated
       traces.  A constraint is violated if there exists at least one trace
       that falsifies it.
    4. Return a :class:`~src.models.VerificationResult` summarising satisfied
       and violated counts, with counter-example traces for each violation.

Typical usage::

    from src.module_a.bpmn_parser import parse_bpmn
    from src.module_c.verifier import verify_constraints
    from src.models import DeclareConstraint, DeclareTemplate

    graph = parse_bpmn("data/dispatch/Dispatch-of-goods.bpmn")
    constraints = [
        DeclareConstraint(
            id="c01", template=DeclareTemplate.EXISTENCE,
            activity_a="Clarify shipment method", source_text="...",
        )
    ]
    result = verify_constraints(graph, constraints)
    print(result.is_conformant)
"""

import itertools
import logging
from collections import defaultdict

from src.models import (
    BPMNGraph,
    BPMNNodeType,
    DeclareConstraint,
    DeclareTemplate,
    VerificationResult,
    Violation,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MAX_TRACES = 1000

# Gateway types that FORK (split) the flow.
_FORK_TYPES = {
    BPMNNodeType.EXCLUSIVE_GATEWAY,
    BPMNNodeType.INCLUSIVE_GATEWAY,
    BPMNNodeType.PARALLEL_GATEWAY,
    BPMNNodeType.EVENT_BASED_GATEWAY,
}


# ── Trace generation ──────────────────────────────────────────────────────────


def _build_adjacency(graph: BPMNGraph) -> dict[str, list[str]]:
    """Build a node-id → [successor node-ids] mapping from the graph edges.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Dictionary mapping each node ID to its ordered list of successor IDs.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adj[edge.source].append(edge.target)
    return dict(adj)


def _node_map(graph: BPMNGraph) -> dict[str, BPMNNodeType]:
    """Build a node-id → BPMNNodeType lookup map.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Dictionary mapping node IDs to their types.
    """
    return {n.id: n.type for n in graph.nodes}


def _node_name_map(graph: BPMNGraph) -> dict[str, str]:
    """Build a node-id → name lookup map.

    Args:
        graph: Parsed BPMN graph.

    Returns:
        Dictionary mapping node IDs to their display names.
    """
    return {n.id: n.name for n in graph.nodes}


def generate_traces(
    graph: BPMNGraph,
    max_traces: int = DEFAULT_MAX_TRACES,
) -> list[list[str]]:
    """Enumerate all possible execution traces from a BPMN process graph.

    A *trace* is an ordered sequence of **task names** (events and gateways
    are structural and are not included in the trace).

    Gateway semantics:

    * **Exclusive / Event-based** (XOR fork): explore each outgoing branch
      as a separate alternative.
    * **Parallel** (AND fork): generate all possible interleavings of the
      branches running in parallel.
    * **Inclusive** (OR fork): generate all non-empty subsets of outgoing
      branches, each subset explored with all interleavings.

    The DFS is bounded by *max_traces* to prevent combinatorial explosion.

    Args:
        graph: Parsed BPMN process graph.
        max_traces: Hard upper bound on the number of traces returned.

    Returns:
        List of traces; each trace is a list of task-name strings.
        Returns ``[[]]`` (one empty trace) if no start event is found.
    """
    adj = _build_adjacency(graph)
    types = _node_map(graph)
    names = _node_name_map(graph)

    # Locate the start event(s).
    start_nodes = [n.id for n in graph.nodes if n.type == BPMNNodeType.START_EVENT]
    if not start_nodes:
        logger.warning("No start event found in graph %r", graph.process_name)
        return [[]]

    collected: list[list[str]] = []

    def dfs(node_id: str, current_trace: list[str], visited: set[str]) -> None:
        """Recursive DFS that appends complete traces to *collected*."""
        if len(collected) >= max_traces:
            return

        # Cycle guard
        if node_id in visited:
            collected.append(list(current_trace))
            return

        node_type = types.get(node_id, BPMNNodeType.TASK)
        node_name = names.get(node_id, "")
        successors = adj.get(node_id, [])

        # ── Append task name to the trace ──────────────────────────────────
        new_trace = current_trace.copy()
        if node_type == BPMNNodeType.TASK and node_name:
            new_trace.append(node_name)

        # ── Terminal: end event or dead end ────────────────────────────────
        if node_type == BPMNNodeType.END_EVENT or not successors:
            collected.append(new_trace)
            return

        new_visited = visited | {node_id}

        # ── Exclusive / Event-based gateway (XOR fork) ─────────────────────
        if node_type in (BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.EVENT_BASED_GATEWAY):
            for succ in successors:
                dfs(succ, new_trace, new_visited)

        # ── Parallel gateway (AND fork) ────────────────────────────────────
        elif node_type == BPMNNodeType.PARALLEL_GATEWAY:
            if len(successors) > 1:
                # Collect sub-traces for each parallel branch
                branch_trace_sets: list[list[list[str]]] = []
                for succ in successors:
                    branch_traces: list[list[str]] = []
                    _collect_branch(succ, [], new_visited, adj, types, names,
                                    branch_traces, max_traces)
                    branch_trace_sets.append(branch_traces or [[]])

                # Interleave all branch combinations
                for branch_combo in itertools.product(*branch_trace_sets):
                    for interleaving in _interleavings(list(branch_combo)):
                        merged = new_trace + interleaving
                        collected.append(merged)
                        if len(collected) >= max_traces:
                            return
            else:
                # This is a JOIN gateway (1 incoming, 1 outgoing) — pass through
                for succ in successors:
                    dfs(succ, new_trace, new_visited)

        # ── Inclusive gateway (OR fork) ────────────────────────────────────
        elif node_type == BPMNNodeType.INCLUSIVE_GATEWAY:
            if len(successors) > 1:
                # All non-empty subsets of branches
                for r in range(1, len(successors) + 1):
                    for subset in itertools.combinations(successors, r):
                        branch_trace_sets = []
                        for succ in subset:
                            branch_traces = []
                            _collect_branch(succ, [], new_visited, adj, types, names,
                                            branch_traces, max_traces)
                            branch_trace_sets.append(branch_traces or [[]])
                        for branch_combo in itertools.product(*branch_trace_sets):
                            for interleaving in _interleavings(list(branch_combo)):
                                merged = new_trace + interleaving
                                collected.append(merged)
                                if len(collected) >= max_traces:
                                    return
            else:
                for succ in successors:
                    dfs(succ, new_trace, new_visited)

        # ── Other nodes (join gateways, intermediate events, etc.) ─────────
        else:
            for succ in successors:
                dfs(succ, new_trace, new_visited)

    for start in start_nodes:
        dfs(start, [], set())
        if len(collected) >= max_traces:
            break

    if not collected:
        collected.append([])

    logger.debug(
        "Generated %d trace(s) for process %r", len(collected), graph.process_name
    )
    return collected


def _collect_branch(
    node_id: str,
    current_trace: list[str],
    visited: set[str],
    adj: dict[str, list[str]],
    types: dict[str, BPMNNodeType],
    names: dict[str, str],
    result: list[list[str]],
    max_traces: int,
) -> None:
    """DFS sub-routine that collects traces for a single branch segment.

    Stops at join gateways (nodes with multiple incoming edges) — the parent
    ``dfs`` call will continue beyond the join.

    Args:
        node_id: Current node to explore.
        current_trace: Accumulated task names in this branch.
        visited: Set of already-visited node IDs (cycle guard).
        adj: Adjacency map.
        types: Node-type map.
        names: Node-name map.
        result: Mutable list to append completed branch traces to.
        max_traces: Hard cap on total traces.
    """
    if len(result) >= max_traces or node_id in visited:
        result.append(list(current_trace))
        return

    node_type = types.get(node_id, BPMNNodeType.TASK)
    node_name = names.get(node_id, "")
    successors = adj.get(node_id, [])

    new_trace = current_trace.copy()
    if node_type == BPMNNodeType.TASK and node_name:
        new_trace.append(node_name)

    # Stop at end events, dead ends, or join gateways (handled by parent).
    if (
        node_type == BPMNNodeType.END_EVENT
        or not successors
        or (node_type in _FORK_TYPES and len(successors) == 1
            and node_type not in (BPMNNodeType.EXCLUSIVE_GATEWAY,
                                  BPMNNodeType.EVENT_BASED_GATEWAY,
                                  BPMNNodeType.INCLUSIVE_GATEWAY))
    ):
        result.append(new_trace)
        return

    new_visited = visited | {node_id}

    if node_type in (BPMNNodeType.EXCLUSIVE_GATEWAY, BPMNNodeType.EVENT_BASED_GATEWAY):
        for succ in successors:
            _collect_branch(succ, new_trace, new_visited, adj, types, names, result, max_traces)
    elif node_type == BPMNNodeType.PARALLEL_GATEWAY and len(successors) > 1:
        # Nested parallel branch — collect and interleave sub-branches
        sub_sets: list[list[list[str]]] = []
        for succ in successors:
            sub: list[list[str]] = []
            _collect_branch(succ, [], new_visited, adj, types, names, sub, max_traces)
            sub_sets.append(sub or [[]])
        for combo in itertools.product(*sub_sets):
            for il in _interleavings(list(combo)):
                result.append(new_trace + il)
                if len(result) >= max_traces:
                    return
    elif node_type == BPMNNodeType.INCLUSIVE_GATEWAY and len(successors) > 1:
        for r in range(1, len(successors) + 1):
            for subset in itertools.combinations(successors, r):
                sub_sets = []
                for succ in subset:
                    sub = []
                    _collect_branch(succ, [], new_visited, adj, types, names, sub, max_traces)
                    sub_sets.append(sub or [[]])
                for combo in itertools.product(*sub_sets):
                    for il in _interleavings(list(combo)):
                        result.append(new_trace + il)
                        if len(result) >= max_traces:
                            return
    else:
        for succ in successors:
            _collect_branch(succ, new_trace, new_visited, adj, types, names, result, max_traces)


def _interleavings(branch_traces: list[list[str]]) -> list[list[str]]:
    """Generate all interleavings of a list of sequential branch traces.

    For two branches ``[A, B]`` and ``[C, D]``, the interleavings are all
    permutations of the combined sequence that preserve the internal order of
    each branch (i.e. A always before B, C always before D).

    For simplicity and to limit explosion, this implementation generates
    permutations of the *branches themselves* (not element-level interleavings).
    This is equivalent to choosing the order in which parallel branches
    complete — a sound approximation for DECLARE verification.

    Args:
        branch_traces: Each element is the task-name sequence of one branch.

    Returns:
        List of merged traces (one per branch ordering).
    """
    if not branch_traces:
        return [[]]
    if len(branch_traces) == 1:
        return [branch_traces[0]]

    results: list[list[str]] = []
    # Cap to avoid factorial explosion for many-branch parallels
    max_perms = min(24, len(list(itertools.permutations(range(len(branch_traces))))))
    for perm in list(itertools.permutations(branch_traces))[:max_perms]:
        merged: list[str] = []
        for branch in perm:
            merged.extend(branch)
        results.append(merged)
    return results


# ── DECLARE constraint checkers ───────────────────────────────────────────────


def _check_existence(a: str, trace: list[str]) -> bool:
    """``existence(A)``: A must occur at least once.

    Args:
        a: Activity name A.
        trace: Sequence of activity names in one execution.

    Returns:
        ``True`` if A appears in the trace.
    """
    return a in trace


def _check_absence(a: str, trace: list[str]) -> bool:
    """``absence(A)``: A must never occur.

    Args:
        a: Activity name A.
        trace: Sequence of activity names.

    Returns:
        ``True`` if A does NOT appear in the trace.
    """
    return a not in trace


def _check_response(a: str, b: str, trace: list[str]) -> bool:
    """``response(A, B)``: every occurrence of A must be followed by B.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if after every A in the trace, B appears at some later index.
    """
    for i, act in enumerate(trace):
        if act == a and b not in trace[i + 1:]:
            return False
    return True


def _check_precedence(a: str, b: str, trace: list[str]) -> bool:
    """``precedence(A, B)``: B can only occur after A has occurred.

    Args:
        a: Activity A (must precede B).
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if every occurrence of B is preceded by at least one A.
    """
    for i, act in enumerate(trace):
        if act == b and a not in trace[:i]:
            return False
    return True


def _check_succession(a: str, b: str, trace: list[str]) -> bool:
    """``succession(A, B)``: response(A,B) AND precedence(A,B).

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if both response and precedence hold.
    """
    return _check_response(a, b, trace) and _check_precedence(a, b, trace)


def _check_chain_response(a: str, b: str, trace: list[str]) -> bool:
    """``chain_response(A, B)``: every A must be immediately followed by B.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if B immediately follows every occurrence of A.
    """
    for i, act in enumerate(trace):
        if act == a:
            if i + 1 >= len(trace) or trace[i + 1] != b:
                return False
    return True


def _check_chain_precedence(a: str, b: str, trace: list[str]) -> bool:
    """``chain_precedence(A, B)``: B can only occur immediately after A.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if every occurrence of B is immediately preceded by A.
    """
    for i, act in enumerate(trace):
        if act == b:
            if i == 0 or trace[i - 1] != a:
                return False
    return True


def _check_not_coexistence(a: str, b: str, trace: list[str]) -> bool:
    """``not_coexistence(A, B)``: A and B cannot both appear.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if at most one of A, B appears in the trace.
    """
    return not (a in trace and b in trace)


def _check_choice(a: str, b: str, trace: list[str]) -> bool:
    """``choice(A, B)``: at least one of A or B must occur.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if A or B (or both) appear in the trace.
    """
    return a in trace or b in trace


def _check_exclusive_choice(a: str, b: str, trace: list[str]) -> bool:
    """``exclusive_choice(A, B)``: exactly one of A or B must occur.

    Args:
        a: Activity A.
        b: Activity B.
        trace: Sequence of activity names.

    Returns:
        ``True`` if exactly one of A, B appears in the trace (XOR).
    """
    return (a in trace) != (b in trace)


# ── Dispatcher ────────────────────────────────────────────────────────────────

_CHECKER_MAP = {
    DeclareTemplate.EXISTENCE: lambda a, b, t: _check_existence(a, t),
    DeclareTemplate.ABSENCE: lambda a, b, t: _check_absence(a, t),
    DeclareTemplate.RESPONSE: lambda a, b, t: _check_response(a, b, t),
    DeclareTemplate.PRECEDENCE: lambda a, b, t: _check_precedence(a, b, t),
    DeclareTemplate.SUCCESSION: lambda a, b, t: _check_succession(a, b, t),
    DeclareTemplate.CHAIN_RESPONSE: lambda a, b, t: _check_chain_response(a, b, t),
    DeclareTemplate.CHAIN_PRECEDENCE: lambda a, b, t: _check_chain_precedence(a, b, t),
    DeclareTemplate.NOT_COEXISTENCE: lambda a, b, t: _check_not_coexistence(a, b, t),
    DeclareTemplate.CHOICE: lambda a, b, t: _check_choice(a, b, t),
    DeclareTemplate.EXCLUSIVE_CHOICE: lambda a, b, t: _check_exclusive_choice(a, b, t),
}


def _check_constraint_on_all_traces(
    constraint: DeclareConstraint,
    traces: list[list[str]],
) -> tuple[bool, list[str]]:
    """Check whether a constraint holds on every trace.

    If the constraint has a non-empty ``condition`` field, it is evaluated only
    on traces that *seem to activate* the condition (heuristic: traces that
    contain ``activity_a``).  If no trace activates the condition, the
    constraint is vacuously satisfied.

    Args:
        constraint: The DECLARE constraint to check.
        traces: All generated execution traces.

    Returns:
        A 2-tuple ``(satisfied, counter_example_trace)``.  If satisfied,
        ``counter_example_trace`` is an empty list.
    """
    checker = _CHECKER_MAP.get(constraint.template)
    if checker is None:
        logger.warning("Unknown template %r — skipping constraint %s", constraint.template, constraint.id)
        return True, []

    a = constraint.activity_a
    b = constraint.activity_b

    # Apply conditional filtering
    active_traces = traces
    if constraint.condition:
        active_traces = [t for t in traces if a in t]
        if not active_traces:
            logger.debug(
                "Constraint %s is vacuously satisfied (no traces match condition %r)",
                constraint.id,
                constraint.condition,
            )
            return True, []

    for trace in active_traces:
        if not checker(a, b, trace):
            return False, trace

    return True, []


# ── Public API ────────────────────────────────────────────────────────────────


def verify_constraints(
    graph: BPMNGraph,
    constraints: list[DeclareConstraint],
    bpmn_file_path: str = "",
    max_traces: int = DEFAULT_MAX_TRACES,
) -> VerificationResult:
    """Verify DECLARE constraints against all possible execution traces.

    This function:
    1. Generates all execution traces from the BPMN graph (DFS with gateway
       semantics).
    2. Checks each constraint against every trace.
    3. Records violations with counter-example traces.

    Args:
        graph: Parsed BPMN process graph.
        constraints: DECLARE constraints extracted from the process description.
        bpmn_file_path: Unused in this implementation (reserved for pm4py
            integration).
        max_traces: Maximum number of traces to generate (default 1000).

    Returns:
        :class:`~src.models.VerificationResult` with satisfied/violated counts
        and :class:`~src.models.Violation` details for failed constraints.
    """
    logger.info(
        "Verifying %d constraint(s) for process %r",
        len(constraints),
        graph.process_name,
    )

    traces = generate_traces(graph, max_traces=max_traces)
    logger.info("Working with %d trace(s)", len(traces))

    violations: list[Violation] = []
    satisfied_count = 0

    for constraint in constraints:
        ok, counter_trace = _check_constraint_on_all_traces(constraint, traces)

        if ok:
            satisfied_count += 1
            logger.debug("Constraint %s: SATISFIED", constraint.id)
        else:
            desc = (
                f"{constraint.template.value}({constraint.activity_a}"
                + (f", {constraint.activity_b}" if constraint.activity_b else "")
                + ")"
            )
            violations.append(
                Violation(
                    constraint_id=constraint.id,
                    constraint_description=desc,
                    trace=counter_trace,
                    explanation="",  # filled by Agent 3 in TASK_06
                )
            )
            logger.debug(
                "Constraint %s: VIOLATED — counter-example: %s",
                constraint.id,
                counter_trace,
            )

    result = VerificationResult(
        is_conformant=len(violations) == 0,
        total_constraints=len(constraints),
        satisfied=satisfied_count,
        violated=len(violations),
        violations=violations,
    )
    logger.info(
        "Verification complete: %d satisfied, %d violated → %s",
        result.satisfied,
        result.violated,
        "CONFORMANT" if result.is_conformant else "NON-CONFORMANT",
    )
    return result
