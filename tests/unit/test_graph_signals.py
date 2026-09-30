"""Unit tests for the "gather data on its own" defaults in src/agents/graph.py:
auto-constructing competitor/macro adapters from configured API keys, and
auto-deriving which competitor brands to search for from the choice set, so a caller
supplying only CSV sales data still gets these signals for free once the relevant
API key is set -- no extra wiring required (see the module docstrings of
`competitor_signal_node` / `macro_signal_node`).
"""

from datetime import date
from pathlib import Path

from config.settings import Settings
from src.agents.graph import _default_macro_source, _default_search_adapter, competitor_signal_node
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.schema.core import AnalysisRequest
from src.schema.tools import DataProfile
from src.signals.competitor import AnthropicWebSearchAdapter, SearchResult, StaticSearchAdapter
from src.signals.macro import FredMacroSource
from src.tools import build_choice_set

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_default_search_adapter_is_none_without_api_key() -> None:
    assert _default_search_adapter(Settings(anthropic_api_key="")) is None


def test_default_search_adapter_constructs_real_adapter_with_api_key() -> None:
    settings = Settings(anthropic_api_key="fake-key", anthropic_model="claude-x")
    adapter = _default_search_adapter(settings)
    assert isinstance(adapter, AnthropicWebSearchAdapter)


def test_default_macro_source_is_none_without_api_key() -> None:
    assert _default_macro_source(Settings(fred_api_key="")) is None


def test_default_macro_source_constructs_real_source_with_api_key() -> None:
    source = _default_macro_source(Settings(fred_api_key="fake-key"))
    assert isinstance(source, FredMacroSource)


def _state_for_competitor_signal_node(search_adapter, competitor_brands=None) -> dict:
    request = AnalysisRequest(
        focal_product_id="1002",
        window=(date(2024, 1, 1), date(2024, 12, 31)),
        regions=["store_1"],
    )
    adapter = CPGScannerAdapter(FIXTURES / "cpg_scanner_sample.csv")
    products = adapter.load_products(request)
    choice_set = build_choice_set(products, request)
    data_profile = DataProfile(
        data_level="aggregate",
        price_variation="sufficient",
        donor_pool="weak",
        pre_periods=1,
        segments_estimable=False,
    )
    return {
        "request": request,
        "products": products,
        "choice_set": choice_set,
        "data_profile": data_profile,
        "search_adapter": search_adapter,
        "competitor_brands": competitor_brands or [],
    }


def test_competitor_signal_node_auto_derives_brands_from_choice_set() -> None:
    """No competitor_brands supplied -- the node should still find the RivalCo event
    because it derives brands from the incumbents relationship_agent already found,
    not require the caller to name them.
    """
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
    state = _state_for_competitor_signal_node(search_adapter, competitor_brands=None)
    result = competitor_signal_node(state)
    assert result["concurrent_shock"].level == "confirmed"
    assert len(result["competitor_events"]) >= 1
    assert any(e.competitor_brand == "RivalCo" for e in result["competitor_events"])


class _RecordingSearchAdapter(StaticSearchAdapter):
    """Records the query strings it was asked to search, so tests can verify *which*
    brands were actually queried without depending on StaticSearchAdapter's
    query-agnostic matching."""

    def __init__(self, results: list[SearchResult]) -> None:
        super().__init__(results)
        self.queries: list[str] = []

    def search(self, query: str, window: tuple[date, date]) -> list[SearchResult]:
        self.queries.append(query)
        return super().search(query, window)


def test_competitor_signal_node_respects_explicit_brands() -> None:
    """An explicitly-supplied brand list is used as-is, not overridden by the
    auto-derived incumbent brands (SnackCo/RivalCo, per the sample CSV)."""
    search_adapter = _RecordingSearchAdapter([])
    state = _state_for_competitor_signal_node(search_adapter, competitor_brands=["SomeOtherBrand"])
    competitor_signal_node(state)

    assert any("SomeOtherBrand" in q for q in search_adapter.queries)
    assert not any("RivalCo" in q for q in search_adapter.queries)


def test_competitor_signal_node_auto_derived_brands_are_the_actual_incumbents() -> None:
    search_adapter = _RecordingSearchAdapter([])
    state = _state_for_competitor_signal_node(search_adapter, competitor_brands=None)
    competitor_signal_node(state)

    # both incumbents' brands (1001=SnackCo, same brand as focal; 1003=RivalCo) --
    # derived straight from the choice set relationship_agent already built, not
    # hand-named by the caller.
    queried_brands = {q.split()[0] for q in search_adapter.queries}
    assert queried_brands == {"SnackCo", "RivalCo"}
