"""Output schema. Every number here must originate from a deterministic tool call in
`src/causal/` -- never from an LLM. See CANNIBALIZATION_AGENT_BUILD_PLAN.md section 1.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Confidence = Literal["high", "medium", "low"]
DominantSource = Literal["within_brand", "competitor", "net_new_demand", "unknown"]
MethodName = Literal[
    "mixed_logit",
    "nested_logit_individual",
    "blp_nested_logit",
    "synthetic_control",
    "causal_impact",
    "did",
    "bayesian_shrinkage_logit",
]


class ConfidenceReport(BaseModel):
    """Uncertainty quantification for a point estimate: CI + the method used to get it."""

    model_config = ConfigDict(frozen=True)

    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float = Field(default=0.95, gt=0, lt=1)
    method: Literal["bootstrap", "delta_method"]
    n_resamples: int | None = None
    seed: int

    @model_validator(mode="after")
    def _ci_bounds_ordered(self) -> ConfidenceReport:
        if not (self.ci_low <= self.point_estimate <= self.ci_high):
            raise ValueError(
                f"point_estimate ({self.point_estimate}) must lie within "
                f"[ci_low={self.ci_low}, ci_high={self.ci_high}]"
            )
        return self


class SegmentResult(BaseModel):
    """Per-segment cannibalization estimate (region/price-tier or estimated customer segment)."""

    model_config = ConfigDict(frozen=True)

    segment_id: str
    cannibalization_rate: float
    ci_low: float
    ci_high: float
    dominant_source: DominantSource = "unknown"


class MassBalanceCheck(BaseModel):
    """Conservation check: removed focal demand must redistribute, not vanish/double-count."""

    model_config = ConfigDict(frozen=True)

    focal_units_removed: float
    redistributed_units_total: float
    tolerance: float = 1e-6

    @property
    def discrepancy(self) -> float:
        return abs(self.focal_units_removed - self.redistributed_units_total)

    @property
    def passed(self) -> bool:
        if self.focal_units_removed == 0:
            return self.discrepancy <= self.tolerance
        return self.discrepancy / abs(self.focal_units_removed) <= self.tolerance


class ReducedFormResult(BaseModel):
    """Output of a reduced-form cross-check (synthetic control / CausalImpact / DiD)."""

    model_config = ConfigDict(frozen=True)

    method: MethodName
    incumbent_product_id: str
    actual_units: float
    counterfactual_units: float
    impact_units: float
    ci_low: float
    ci_high: float
    pre_period_fit_score: float | None = None
    parallel_trends_passed: bool | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)


class ValidatorFlag(BaseModel):
    """A single hard-gate finding recorded by the validator node."""

    model_config = ConfigDict(frozen=True)

    check: str
    passed: bool
    detail: str = ""
    severity: Literal["info", "warning", "blocking"] = "warning"


class CannibalizationResult(BaseModel):
    """The final, reportable estimate. Numbers here are traceable to tool outputs only."""

    model_config = ConfigDict(frozen=True)

    focal_product_id: str
    overall_rate: float
    ci_low: float
    ci_high: float
    segments: list[SegmentResult] = Field(default_factory=list)
    method_used: MethodName
    cross_check_methods: list[MethodName] = Field(default_factory=list)
    confidence: Confidence
    confounders_flagged: list[str] = Field(default_factory=list)
    validator_flags: list[ValidatorFlag] = Field(default_factory=list)
    mass_balance: MassBalanceCheck | None = None
    seed: int
    narrative: str = ""
    halted: bool = False
    halt_reason: str | None = None
