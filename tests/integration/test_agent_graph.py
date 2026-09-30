"""Phase 3/4 acceptance (build plan sections 8-9): the full graph runs end-to-end on
fixture data, and competitor/macro signals -- when supplied -- are actually folded
into the fitted model's math (not just used to flag a confound afterward), measurably
changing the estimate. `tests/unit/test_validator.py` covers the hard-gate halt path
directly (a confirmed shock that truly *couldn't* be modeled as a covariate still
blocks output); this file covers the case that motivated wiring it in: a confirmed
shock that *was* modeled gets a lower confidence label instead of being thrown away.

Uses `run_pipeline_sync` (see `graph.py` docstring) so this doesn't require the
`agents`/LangGraph extra to be installed -- only the graph *wiring* (node order,
state flow) is under test here, not LangGraph itself.
"""

from datetime import date
from pathlib import Path

from src.agents.graph import run_pipeline_sync
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.schema.core import AnalysisRequest
from src.signals.competitor import SearchResult, StaticSearchAdapter
from src.signals.macro import StaticMacroSource

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _request() -> AnalysisRequest:
    # regions pinned to the sample CSV's actual region code ("store_1") -- both
    # macro_signal_node and competitor_signal_node fetch for request.regions[0], and
    # build_market_context joins signals back to sales by that same region code.
    return AnalysisRequest(
        focal_product_id="1002",
        window=(date(2024, 1, 1), date(2024, 12, 31)),
        regions=["store_1"],
    )


def test_pipeline_runs_end_to_end_and_numbers_trace_to_tools() -> None:
    adapter = CPGScannerAdapter(FIXTURES / "cpg_scanner_sample.csv")
    result = run_pipeline_sync(adapter, _request())

    assert result.halted is False
    assert result.focal_product_id == "1002"
    assert 0.0 <= result.overall_rate <= 1.0
    assert result.ci_low <= result.overall_rate <= result.ci_high
    assert result.method_used == "blp_nested_logit"
    assert result.mass_balance is not None and result.mass_balance.passed
    assert result.narrative  # reporter populated it
    assert result.confidence in ("high", "medium", "low")


def test_confirmed_competitor_shock_is_folded_into_math_and_downgrades_confidence() -> None:
    adapter = CPGScannerAdapter(FIXTURES / "cpg_scanner_sample.csv")
    search_adapter = StaticSearchAdapter(
        [
            SearchResult(
                title="RivalCo launches aggressive new chip line",
                snippet="RivalCo today announced the launch of a competing product",
                url="http://example.com/1",
                published=date(2024, 1, 20),
                source="news",
            )
        ]
    )
    baseline = run_pipeline_sync(adapter, _request())
    result = run_pipeline_sync(
        adapter, _request(), search_adapter=search_adapter, competitor_brands=["RivalCo"]
    )

    # Modeled, not thrown away: confidence drops but the pipeline still reports a number.
    assert result.halted is False
    assert result.confidence == "low"
    assert "confirmed_competitor_or_macro_shock" in result.confounders_flagged
    assert result.narrative

    # The whole point of wiring this in: the competitor covariate actually changes the
    # fitted model, so the estimate differs from the baseline run without it.
    assert result.overall_rate != baseline.overall_rate


def test_macro_data_measurably_changes_the_estimate() -> None:
    adapter = CPGScannerAdapter(FIXTURES / "cpg_scanner_sample.csv")
    # a macro index that differs sharply between the two observed weeks
    macro_source = StaticMacroSource(
        {
            ("cpi", "store_1"): {
                date(2024, 1, 1): 250.0,
                date(2024, 2, 1): 400.0,
            }
        }
    )
    baseline = run_pipeline_sync(adapter, _request())
    result = run_pipeline_sync(
        adapter, _request(), macro_source=macro_source, macro_indicators=["cpi"]
    )

    assert result.halted is False
    assert result.overall_rate != baseline.overall_rate
