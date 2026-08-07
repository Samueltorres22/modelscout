"""LangGraph wiring: fan-out one process_model_node call per candidate,
converge on notifier_node.

Earlier version of this file tried to model Triage and Extractor/Skip as
SEPARATE graph nodes connected by a conditional edge, on the theory that the
cost-aware-routing guarantee should be a structural graph property (no edge
into extractor_node except from the "extract" branch) rather than an `if`
buried in a function. That version does not work: LangGraph's Send-based
fan-out means every parallel branch shares the SAME graph-level state
channels, and any state key without an Annotated reducer (triage_label,
is_relevant, ...) raised `InvalidUpdateError: Can receive only one value per
step` the moment more than one candidate ran in the same superstep --
verified empirically, not a hypothetical. Each Send-spawned branch does NOT
get an isolated private state; only Annotated-reducer keys (here, `results`)
are safe to write from multiple parallel branches.

Fix: collapse the per-item chain into ONE node (process_model_node) that
internally calls the same triage_node/extractor_node/skip_node FUNCTIONS
(still independently unit-tested, still reusable) and returns a single
`{"results": [...]}` write -- the only channel multiple parallel branches
ever touch, and the one channel that has an operator.add reducer. The
cost-aware-routing guarantee is now enforced by the explicit if/else below
rather than by graph topology; it's still concretely checked (not just
asserted in a docstring) via the `n_extracted == n_triage_pass` assertion in
pipeline.py after every run.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from modelscout.agents.extractor_node import extractor_node, skip_node
from modelscout.agents.notifier_node import notifier_node
from modelscout.agents.state import ModelState, PipelineState
from modelscout.agents.triage_node import triage_node


def fan_out_to_process(state: PipelineState) -> list[Send]:
    profile = state["interest_profile"]
    return [
        Send(
            "process_model_node",
            {
                "model_id": c["model_id"],
                "readme_text": c["readme_text"],
                "downloads": c.get("downloads", 0),
                "tags": c.get("tags", []),
                "interest_profile": profile,
            },
        )
        for c in state["candidates"]
    ]


def process_model_node(state: ModelState) -> dict:
    triage_update = triage_node(state)
    merged: ModelState = {**state, **triage_update}  # type: ignore[assignment]

    if merged["is_relevant"]:
        return extractor_node(merged)  # the only path that calls the Anthropic API
    return skip_node(merged)


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("process_model_node", process_model_node)
    graph.add_node("notifier_node", notifier_node)

    graph.add_conditional_edges(START, fan_out_to_process, ["process_model_node"])
    graph.add_edge("process_model_node", "notifier_node")
    graph.add_edge("notifier_node", END)

    return graph.compile()
