"""The LangGraph DAG (build plan section 8): a typed-state graph with conditional
edges, not a free-form chat loop. `langgraph` is imported lazily inside
`build_graph()` so the rest of `src/agents/` (and its unit tests) work without the
`agents` extra installed -- only running the compiled graph needs it.

    planner -> data_agent -> relationship_agent -> method_selector -> causal_agent
        -> validator -> {reporter | END (halted)}
"""

from __future__ import annotations

from config.settings import Settings, get_settings
from src.agents.causal_agent import run_causal_analysis
from src.agents.data_agent import load_and_profile
from src.agents.method_selector import select_method
from src.agents.planner import parse_request
from src.agents.relationship_agent import build_relationships
from src.agents.reporter import narrate
from src.agents.state import PipelineState
from src.agents.validator import validate
from src.schema.signals import ConcurrentShockAssessment
from src.signals.competitor import assess_concurrent_shock, fetch_competitor_events


def planner_node(state: PipelineState) -> dict:
    if "request" in state:
        return {}  # caller already supplied a structured request; skip NL parsing
    settings = get_settings()
    request = parse_request(
        state["raw_query"], api_key=settings.anthropic_api_key, model=settings.anthropic_model
    )
    return {"request": request}


def data_agent_node(state: PipelineState) -> dict:
    products, sales, profile = load_and_profile(state["adapter"], state["request"])
    return {"products": products, "sales": sales, "data_profile": profile}


def relationship_agent_node(state: PipelineState) -> dict:
    choice_set = build_relationships(state["products"], state["request"])
    return {"choice_set": choice_set, "candidate_incumbent_ids": choice_set.candidate_incumbent_ids}


def competitor_signal_node(state: PipelineState) -> dict:
    """Optional Phase 4 step: only runs when the caller supplied a `search_adapter` +
    `competitor_brands`; otherwise `concurrent_shock` defaults to "none" and the
    validator's override simply doesn't fire.
    """
    search_adapter = state.get("search_adapter")
    brands = state.get("competitor_brands") or []
    if search_adapter is None or not brands:
        assessment = ConcurrentShockAssessment(
            level="none", rationale="No competitor signal source configured."
        )
        return {"concurrent_shock": assessment, "data_profile": state["data_profile"]}

    focal = next((p for p in state["products"] if p.is_focal), None)
    category = focal.category if focal else ""
    region = state["request"].regions[0] if state["request"].regions else ""
    events = fetch_competitor_events(
        search_adapter, state["request"].window, category, region, brands
    )
    assessment = assess_concurrent_shock(events, covariate_included=False)
    updated_profile = state["data_profile"].model_copy(
        update={"concurrent_shock": assessment.level}
    )
    return {"concurrent_shock": assessment, "data_profile": updated_profile}


def method_selector_node(state: PipelineState) -> dict:
    selection = select_method(state["data_profile"])
    return {"method_selection": selection}


def causal_agent_node(state: PipelineState) -> dict:
    settings = get_settings()
    result = run_causal_analysis(
        state["products"],
        state["sales"],
        state["choice_set"],
        state["method_selection"],
        seed=settings.random_seed,
    )
    return {"causal_agent_result": result}


def validator_node(state: PipelineState) -> dict:
    causal_result = state["causal_agent_result"]
    concurrent_shock = state.get("concurrent_shock") or ConcurrentShockAssessment(level="none")
    final = validate(causal_result, state["method_selection"], concurrent_shock)
    return {
        "final_result": final,
        "halted": final.halted,
        "halt_reason": final.halt_reason,
        "confidence": final.confidence,
        "validator_flags": final.validator_flags,
    }


def route_after_validator(state: PipelineState) -> str:
    return "halt" if state.get("halted") else "reporter"


def reporter_node(state: PipelineState) -> dict:
    settings = get_settings()
    narrated = narrate(
        state["final_result"],
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_report_model,
    )
    return {"final_result": narrated}


def halt_node(state: PipelineState) -> dict:
    narrated = narrate(
        state["final_result"]
    )  # deterministic template, no API key needed to explain a halt
    return {"final_result": narrated}


def build_graph(settings: Settings | None = None):  # noqa: ANN201 -- return type is langgraph's CompiledGraph
    """Compiles the pipeline graph. Requires the `agents` extra
    (`uv pip install '.[agents]'`)."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("relationship_agent", relationship_agent_node)
    graph.add_node("competitor_signal", competitor_signal_node)
    graph.add_node("method_selector", method_selector_node)
    graph.add_node("causal_agent", causal_agent_node)
    graph.add_node("validator", validator_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("halt", halt_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "data_agent")
    graph.add_edge("data_agent", "relationship_agent")
    graph.add_edge("relationship_agent", "competitor_signal")
    graph.add_edge("competitor_signal", "method_selector")
    graph.add_edge("method_selector", "causal_agent")
    graph.add_edge("causal_agent", "validator")
    graph.add_conditional_edges(
        "validator", route_after_validator, {"reporter": "reporter", "halt": "halt"}
    )
    graph.add_edge("reporter", END)
    graph.add_edge("halt", END)
    return graph.compile()


def run_pipeline_sync(
    adapter,  # DataAdapter
    request,  # AnalysisRequest
    search_adapter=None,
    competitor_brands: list[str] | None = None,
):
    """Convenience entry point used by the FastAPI layer and tests: runs every node
    as plain function calls, without requiring `langgraph` to be installed. Produces
    the identical result the compiled graph would (same node functions, same order) --
    `build_graph()` above is the LangGraph-native version for when tracing/checkpointing
    (LangSmith) is wanted.
    """
    state: PipelineState = {
        "adapter": adapter,
        "request": request,
        "search_adapter": search_adapter,
        "competitor_brands": competitor_brands or [],
    }
    for node in (
        planner_node,
        data_agent_node,
        relationship_agent_node,
        competitor_signal_node,
        method_selector_node,
        causal_agent_node,
        validator_node,
    ):
        state.update(node(state))  # type: ignore[typeddict-item]

    state.update((halt_node if state.get("halted") else reporter_node)(state))  # type: ignore[typeddict-item]
    return state["final_result"]
