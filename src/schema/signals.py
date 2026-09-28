"""External-signal schema: competitor events, macro series, loyalty profiles.

These feed IN as covariates / donor-selection criteria / utility terms -- the causal
engine re-estimates with them, it never annotates a result with them after the fact.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal[
    "competitor_launch", "price_cut", "price_increase", "stockout", "promotion", "other"
]
ShockLevel = Literal["none", "suspected", "confirmed"]


class CompetitorEvent(BaseModel):
    """A sourced competitor move (launch, price cut, stockout, ...) in the analysis window."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    competitor_brand: str
    category: str
    region: str
    event_date: date
    kind: EventKind
    magnitude: float | None = None
    source: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    summary: str = ""


class MacroSeries(BaseModel):
    """A region's macro indicator time series over the analysis window."""

    model_config = ConfigDict(frozen=True)

    region: str
    indicator: str  # e.g. "cpi", "fx_rate", "disposable_income"
    series: dict[date, float] = Field(default_factory=dict)
    source: str = "FRED"


class LoyaltyProfile(BaseModel):
    """Brand-loyalty summary for a segment or population, used in nest structure / utility."""

    model_config = ConfigDict(frozen=True)

    brand: str
    segment_id: str | None = None
    loyalty_score: float = Field(ge=0, le=1)
    repeat_purchase_rate: float | None = Field(default=None, ge=0, le=1)
    tenure_months: float | None = None


class ConcurrentShockAssessment(BaseModel):
    """The competitor/macro agents' joint verdict on whether a confound overlaps the window."""

    model_config = ConfigDict(frozen=True)

    level: ShockLevel
    events: list[CompetitorEvent] = Field(default_factory=list)
    covariate_included: bool = False
    rationale: str = ""
