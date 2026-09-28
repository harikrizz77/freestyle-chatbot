"""Typed I/O contracts for the tools layer (build plan section 7). Every tool function
in `src/agents/*_agent.py` and the LangGraph nodes takes and returns one of these --
never a raw dict -- so the graph state stays fully typed end to end.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schema.core import CustomerSegment, Product
from src.schema.results import ConfidenceReport, MethodName, ReducedFormResult
from src.schema.signals import CompetitorEvent, MacroSeries

DataLevel = Literal["transaction", "panel", "aggregate"]
PriceVariation = Literal["none", "low", "sufficient"]
DonorPoolStrength = Literal["none", "weak", "strong"]
ConcurrentShockLevel = Literal["none", "suspected", "confirmed"]


class ChoiceSetEntry(BaseModel):
    """One alternative in the built choice set, with its substitutability score
    relative to the focal product (embedding/attribute similarity, domain-agnostic)."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    nest: str
    substitutability_to_focal: float = Field(ge=0, le=1)
    is_focal: bool = False


class ChoiceSet(BaseModel):
    """The full choice set (including the no-purchase option) the causal engine will
    reason over, plus the substitutability scores used to assign/validate nests."""

    model_config = ConfigDict(frozen=True)

    focal_product_id: str
    entries: list[ChoiceSetEntry]
    candidate_incumbent_ids: list[str]


class DataProfile(BaseModel):
    """The fact sheet `data_agent` computes and `method_selector` consumes verbatim
    from `config/method_rules.yaml` -- see that file's `inputs` section. The LLM never
    invents these facts; they come from deterministic checks on the loaded data.
    """

    model_config = ConfigDict(frozen=True)

    data_level: DataLevel
    price_variation: PriceVariation
    donor_pool: DonorPoolStrength
    pre_periods: int
    segments_estimable: bool
    concurrent_shock: ConcurrentShockLevel = "none"
    n_products: int = 0
    n_regions: int = 0
    coverage_notes: str = ""


class MethodSelection(BaseModel):
    """The method_selector's output: which rule fired, and what it prescribes."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    primary_method: MethodName
    cross_check_methods: list[MethodName]
    segmentation: str
    confidence_ceiling: Literal["high", "medium", "low"] | None = None
    notes: str = ""


class SegmentationResult(BaseModel):
    """Output of `segment_customers`: either hard segments or a note that only
    region/price-tier segmentation is available (no panel data)."""

    model_config = ConfigDict(frozen=True)

    segments: list[CustomerSegment]
    method: Literal["clustering", "mixed_logit_distributional", "region_or_price_tier", "none"]


class CompetitorSignalBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: list[CompetitorEvent]
    shock_level: ConcurrentShockLevel
    covariate_included: bool = False


class MacroSignalBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    series: list[MacroSeries]


class CrossCheckComparison(BaseModel):
    """Validator input: primary vs. cross-check reduced-form results, and whether they
    agree within `method_rules.yaml`'s `cross_check_agreement_tolerance`."""

    model_config = ConfigDict(frozen=True)

    primary: ReducedFormResult
    cross_checks: list[ReducedFormResult]
    disagreement: bool
    tolerance: float


class ProductCatalog(BaseModel):
    """Products keyed for convenient lookup across the agent graph."""

    model_config = ConfigDict(frozen=True)

    products: list[Product]

    def get(self, product_id: str) -> Product | None:
        return next((p for p in self.products if p.product_id == product_id), None)

    def focal(self) -> Product | None:
        return next((p for p in self.products if p.is_focal), None)


__all__ = [
    "ChoiceSet",
    "ChoiceSetEntry",
    "CompetitorSignalBundle",
    "ConfidenceReport",
    "CrossCheckComparison",
    "DataProfile",
    "MacroSignalBundle",
    "MethodSelection",
    "ProductCatalog",
    "SegmentationResult",
]
