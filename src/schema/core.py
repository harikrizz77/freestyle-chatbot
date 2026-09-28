"""The normalized data contract. Every DataAdapter maps raw, industry-specific data
into these models. All causal and agent code operates ONLY on these types -- industry
specifics live exclusively in `src/data/adapters/`.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Granularity = Literal["week", "month"]


class Product(BaseModel):
    """A single product/SKU in the analysis, focal or incumbent."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    name: str
    category: str
    brand: str
    attributes: dict[str, float | str] = Field(default_factory=dict)
    price_tier: str
    launch_date: date | None = None
    is_focal: bool = False
    nest: str | None = None


class SalesObservation(BaseModel):
    """One product x region x period sales record."""

    model_config = ConfigDict(frozen=True)

    product_id: str
    region: str
    period: date
    units: float = Field(ge=0)
    revenue: float | None = Field(default=None, ge=0)
    price: float = Field(ge=0)
    promo_flag: bool = False
    market_size: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _market_size_covers_units(self) -> SalesObservation:
        if self.market_size is not None and self.units > self.market_size:
            raise ValueError(
                f"units ({self.units}) cannot exceed market_size ({self.market_size}) "
                f"for product={self.product_id} region={self.region} period={self.period}"
            )
        return self


class MarketContext(BaseModel):
    """Region x period external context: macro indices and competitor pressure."""

    model_config = ConfigDict(frozen=True)

    region: str
    period: date
    macro_index: dict[str, float] = Field(default_factory=dict)
    competitor_pressure: dict[str, float] = Field(default_factory=dict)


class CustomerSegment(BaseModel):
    """A derived customer segment. Only populated when panel/transaction data exists."""

    model_config = ConfigDict(frozen=True)

    segment_id: str
    loyalty_score: float = Field(ge=0, le=1)
    price_sensitivity: float
    income_proxy: float
    size: float = Field(ge=0)


class AnalysisRequest(BaseModel):
    """The parsed intent of an analysis: what to estimate, over what window/regions."""

    model_config = ConfigDict(frozen=True)

    focal_product_id: str
    candidate_incumbents: list[str] | None = None
    window: tuple[date, date]
    regions: list[str] | None = None
    granularity: Granularity = "week"

    @model_validator(mode="after")
    def _window_is_ordered(self) -> AnalysisRequest:
        start, end = self.window
        if start >= end:
            raise ValueError(f"window start ({start}) must be before end ({end})")
        return self
