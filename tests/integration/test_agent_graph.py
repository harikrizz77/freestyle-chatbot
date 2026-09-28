"""Phase 3 acceptance (build plan section 8): the full graph runs end-to-end on
fixture data, and the validator provably blocks a seeded bad case (a confirmed,
un-modeled concurrent competitor launch) rather than emitting a number over it.
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

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _request() -> AnalysisRequest:
    return AnalysisRequest(focal_product_id="1002", window=(date(2024, 1, 1), date(2024, 12, 31)))


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


def test_validator_blocks_confirmed_unmodeled_competitor_shock() -> None:
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
    result = run_pipeline_sync(
        adapter, _request(), search_adapter=search_adapter, competitor_brands=["RivalCo"]
    )

    assert result.halted is True
    assert result.halt_reason is not None
    assert "competitor" in result.halt_reason.lower() or "macro" in result.halt_reason.lower()
    assert result.confidence == "low"
    assert "confirmed_competitor_or_macro_shock_not_modeled" in result.confounders_flagged
    assert result.narrative  # halt_node still narrates *why* it stopped
