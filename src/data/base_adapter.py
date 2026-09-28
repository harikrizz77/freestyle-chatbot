"""The `DataAdapter` contract: new industry = new adapter, zero changes to causal code.
This is the architecture's core agnosticism proof (build plan section 7.3 / 12).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema.core import AnalysisRequest, MarketContext, Product, SalesObservation


class DataAdapter(ABC):
    """Base class every industry-specific data source implements. `load()` is the only
    required method: raw, industry-specific data in, normalized schema out. Nothing
    downstream of this class may know anything about the source format.
    """

    source_name: str

    @abstractmethod
    def load_products(self, request: AnalysisRequest) -> list[Product]:
        """Return the focal product and candidate incumbents/competitors in scope."""

    @abstractmethod
    def load_sales(self, request: AnalysisRequest) -> list[SalesObservation]:
        """Return normalized sales observations covering `request.window`/`regions`."""

    def load_market_context(self, request: AnalysisRequest) -> list[MarketContext]:
        """Optional: region x period macro/competitor context. Default: none available;
        the signals layer (Phase 4) fills this in from FRED/World Bank/news adapters
        instead when a source doesn't carry its own context.
        """
        return []

    def data_level(self) -> str:
        """One of "transaction", "panel", "aggregate" -- the finest granularity this
        adapter can provide, consumed by `method_selector` via `method_rules.yaml`.
        """
        return "aggregate"
