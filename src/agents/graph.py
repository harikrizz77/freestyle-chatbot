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
from src.signals.competitor import (
    AnthropicWebSearchAdapter,
    SearchAdapter,
    assess_concurrent_shock,
    fetch_competitor_events,
)
from src.signals.macro import FredMacroSource, MacroSource, fetch_macro
from src.tools import build_market_context

# Fetched automatically when a `macro_source` is active (explicit or auto-constructed
# from `settings.fred_api_key`) but the caller didn't name specific indicators.
DEFAULT_MACRO_INDICATORS = ["cpi", "disposable_income"]


def _default_search_adapter(settings: Settings) -> SearchAdapter | None:
    """The "gather data on its own" default for competitor signals: if the user has
    configured an Anthropic API key, use the real web-search adapter automatically --
    no extra wiring required. Returns None (competitor signals stay off) otherwise.
    A separate function so tests can monkeypatch it instead of hitting the real API.
    """
    if not settings.anthropic_api_key:
        return None
    return AnthropicWebSearchAdapter(settings.anthropic_api_key, settings.anthropic_model)


def _default_macro_source(settings: Settings) -> MacroSource | None:
    """The "gather data on its own" default for macro signals: if the user has
    configured a FRED API key, use it automatically. Returns None (macro signals stay
    off) otherwise. World Bank isn't auto-selected here since it needs an ISO3 country
    code we can't reliably infer from an arbitrary store/region string -- pass
    `macro_source=WorldBankMacroSource(...)` explicitly for non-US analyses.
    """
    if not settings.fred_api_key:
        return None
    return FredMacroSource(settings.fred_api_key)


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
    """Phase 4 step. Fully automatic when an Anthropic API key is configured: with no
    extra input, this auto-constructs a real web-search adapter (`_default_search_adapter`)
    and auto-derives which brands to search for from the choice set the relationship
    agent just built (the incumbents this analysis is actually comparing against) --
    "gather data on its own," CSV/sales data aside. An explicitly-supplied
    `search_adapter`/`competitor_brands` always takes precedence. With neither
    configured nor supplied, `concurrent_shock` defaults to "none" and the validator's
    override simply doesn't fire. `covariate_included` is left False here --
    `causal_agent_node` sets it once it knows whether the chosen method actually has a
    covariate slot for this data (see `build_market_context`).
    """
    settings = get_settings()
    search_adapter = state.get("search_adapter") or _default_search_adapter(settings)
    brands = state.get("competitor_brands") or []
    if not brands and search_adapter is not None:
        incumbent_ids = set(state["choice_set"].candidate_incumbent_ids)
        brands = sorted({p.brand for p in state["products"] if p.product_id in incumbent_ids})

    if search_adapter is None or not brands:
        assessment = ConcurrentShockAssessment(
            level="none", rationale="No competitor signal source configured."
        )
        return {
            "concurrent_shock": assessment,
            "competitor_events": [],
            "data_profile": state["data_profile"],
        }

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
    return {
        "concurrent_shock": assessment,
        "competitor_events": events,
        "data_profile": updated_profile,
    }


def macro_signal_node(state: PipelineState) -> dict:
    """Phase 4 step. Fully automatic when a FRED API key is configured: with no extra
    input, this auto-constructs `FredMacroSource` (`_default_macro_source`) and
    defaults to `DEFAULT_MACRO_INDICATORS`. An explicitly-supplied
    `macro_source`/`macro_indicators` always takes precedence. With neither configured
    nor supplied, `macro_series` is empty and macro data simply doesn't appear in
    `market_context` below.
    """
    settings = get_settings()
    macro_source = state.get("macro_source") or _default_macro_source(settings)
    indicators = state.get("macro_indicators") or (
        DEFAULT_MACRO_INDICATORS if macro_source is not None else []
    )
    if macro_source is None or not indicators:
        return {"macro_series": []}

    region = state["request"].regions[0] if state["request"].regions else ""
    series = fetch_macro(macro_source, region, state["request"].window, indicators)
    return {"macro_series": series}


def method_selector_node(state: PipelineState) -> dict:
    selection = select_method(state["data_profile"])
    return {"method_selection": selection}


def causal_agent_node(state: PipelineState) -> dict:
    settings = get_settings()
    competitor_events = state.get("competitor_events") or []
    macro_series = state.get("macro_series") or []
    market_context = (
        build_market_context(state["sales"], competitor_events, macro_series)
        if competitor_events or macro_series
        else []
    )
    result = run_causal_analysis(
        state["products"],
        state["sales"],
        state["choice_set"],
        state["method_selection"],
        seed=settings.random_seed,
        market_context=market_context or None,
    )
    # Now that we know whether the chosen method actually folded this data into its
    # math, finalize covariate_included -- this is what lets the validator downgrade
    # a confirmed shock to "low confidence" instead of halting outright.
    concurrent_shock = state.get("concurrent_shock")
    if concurrent_shock is not None:
        concurrent_shock = concurrent_shock.model_copy(
            update={"covariate_included": result.used_market_context}
        )
    return {
        "causal_agent_result": result,
        "market_context": market_context,
        "concurrent_shock": concurrent_shock,
    }


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
        data_profile=state.get("data_profile"),
        method_selection=state.get("method_selection"),
    )
    return {"final_result": narrated}


def halt_node(state: PipelineState) -> dict:
    # deterministic template, no API key needed to explain a halt
    narrated = narrate(
        state["final_result"],
        data_profile=state.get("data_profile"),
        method_selection=state.get("method_selection"),
    )
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
    graph.add_node("macro_signal", macro_signal_node)
    graph.add_node("method_selector", method_selector_node)
    graph.add_node("causal_agent", causal_agent_node)
    graph.add_node("validator", validator_node)
    graph.add_node("reporter", reporter_node)
    graph.add_node("halt", halt_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "data_agent")
    graph.add_edge("data_agent", "relationship_agent")
    graph.add_edge("relationship_agent", "competitor_signal")
    graph.add_edge("competitor_signal", "macro_signal")
    graph.add_edge("macro_signal", "method_selector")
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
    macro_source=None,
    macro_indicators: list[str] | None = None,
):
    """Convenience entry point used by the FastAPI layer and tests: runs every node
    as plain function calls, without requiring `langgraph` to be installed. Produces
    the identical result the compiled graph would (same node functions, same order) --
    `build_graph()` above is the LangGraph-native version for when tracing/checkpointing
    (LangSmith) is wanted.

    `search_adapter`/`competitor_brands` and `macro_source`/`macro_indicators` are both
    optional and independent: supply either, both, or neither. Whatever is supplied is
    actually folded into the fitted model's covariates (see `build_market_context` and
    `src.tools.build_aggregate_design`), not just used to flag a confound afterward.
    """
    state: PipelineState = {
        "adapter": adapter,
        "request": request,
        "search_adapter": search_adapter,
        "competitor_brands": competitor_brands or [],
        "macro_source": macro_source,
        "macro_indicators": macro_indicators or [],
    }
    for node in (
        planner_node,
        data_agent_node,
        relationship_agent_node,
        competitor_signal_node,
        macro_signal_node,
        method_selector_node,
        causal_agent_node,
        validator_node,
    ):
        state.update(node(state))  # type: ignore[typeddict-item]

    state.update((halt_node if state.get("halted") else reporter_node)(state))  # type: ignore[typeddict-item]
    return state["final_result"]
