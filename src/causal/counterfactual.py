"""Two-world counterfactual construction + share redistribution.

This is the mathematical heart of the system (build plan section 6.2). Given a fitted
structural choice model, we compare:

  * Factual world  -- the focal product's utility is present (normal choice set).
  * Counterfactual world -- the focal product's utility is masked to -inf (removed).

The nested-logit formula in `nested_logit.py` handles "within-nest first, then across
nests" redistribution automatically: because nesting correlates alternatives sharing a
nest more tightly (governed by lambda), removing focal's utility and renormalizing
naturally sends most of its mass to same-nest (within-brand) products first, with the
remainder spread across nests (competitors / no-purchase) according to lambda.

Every quantity here is exact (not simulated), so the mass-balance identity holds to
floating-point precision: the focal's factual purchase probability equals exactly the
sum of probability gains everywhere else, because probabilities sum to 1 in both
worlds. `uncertainty.mass_balance_check` re-verifies this as a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.causal.nested_logit import NO_PURCHASE_ID, FittedChoiceModel
from src.schema.results import DominantSource, MassBalanceCheck, SegmentResult


@dataclass(frozen=True)
class SegmentRedistribution:
    segment_id: str
    focal_units: float
    within_brand_units: float
    competitor_units: float
    net_new_units: float
    incumbent_units: dict[str, float]  # product_id -> cannibalized units, this segment

    @property
    def dominant_source(self) -> DominantSource:
        buckets: dict[DominantSource, float] = {
            "within_brand": self.within_brand_units,
            "competitor": self.competitor_units,
            "net_new_demand": self.net_new_units,
        }
        if all(v <= 0 for v in buckets.values()):
            return "unknown"
        return max(buckets, key=lambda k: buckets[k])


@dataclass(frozen=True)
class CounterfactualOutcome:
    focal_product_id: str
    total_focal_units: float
    incumbent_units: dict[str, float]  # product_id -> total cannibalized units (all segments)
    segments: tuple[SegmentRedistribution, ...]
    mass_balance: MassBalanceCheck

    def cannibalization_rate(self, incumbent_id: str) -> float:
        if self.total_focal_units <= 0:
            return 0.0
        return self.incumbent_units.get(incumbent_id, 0.0) / self.total_focal_units

    def overall_cannibalization_rate(self, incumbent_ids: list[str]) -> float:
        if self.total_focal_units <= 0:
            return 0.0
        return (
            sum(self.incumbent_units.get(pid, 0.0) for pid in incumbent_ids)
            / self.total_focal_units
        )

    def segment_results(self, incumbent_ids: list[str]) -> list[SegmentResult]:
        out = []
        for seg in self.segments:
            rate = 0.0
            if seg.focal_units > 0:
                rate = (
                    sum(seg.incumbent_units.get(pid, 0.0) for pid in incumbent_ids)
                    / seg.focal_units
                )
            out.append(
                SegmentResult(
                    segment_id=seg.segment_id,
                    cannibalization_rate=rate,
                    ci_low=rate,
                    ci_high=rate,
                    dominant_source=seg.dominant_source,
                )
            )
        return out


def compute_counterfactual(
    fitted: FittedChoiceModel,
    focal_product_id: str,
    mass_balance_tolerance: float = 1e-6,
) -> CounterfactualOutcome:
    """Run both worlds and redistribute focal's mass. Pure function of `fitted` --
    no I/O, no randomness, fully unit-testable against a hand-computed fixture.
    """
    design = fitted.design
    if focal_product_id not in design.product_ids:
        raise ValueError(f"focal_product_id {focal_product_id!r} not in design.product_ids")

    focal_idx = design.product_index(focal_product_id)
    focal_nest = design.nest_of_product[focal_product_id]

    p_factual = fitted.choice_probabilities()
    p_counterfactual = fitted.choice_probabilities(masked_products=frozenset({focal_product_id}))

    delta = p_counterfactual - p_factual  # (N, J); negative at focal, >=0 elsewhere (typically)
    units_delta = delta * design.market_size[:, None]  # (N, J)

    focal_units_per_obs = p_factual[:, focal_idx] * design.market_size
    total_focal_units = float(focal_units_per_obs.sum())

    incumbent_units: dict[str, float] = {}
    for j, pid in enumerate(design.product_ids):
        if pid == focal_product_id:
            continue
        incumbent_units[pid] = float(units_delta[:, j].sum())

    segments: list[SegmentRedistribution] = []
    seg_ids_unique = sorted(set(design.segment_ids))
    for seg_id in seg_ids_unique:
        rows = [i for i, s in enumerate(design.segment_ids) if s == seg_id]
        seg_focal_units = float(focal_units_per_obs[rows].sum())
        seg_incumbent_units: dict[str, float] = {}
        within_brand = 0.0
        competitor = 0.0
        net_new = 0.0
        for j, pid in enumerate(design.product_ids):
            if pid == focal_product_id:
                continue
            gained = float(units_delta[rows, j].sum())
            seg_incumbent_units[pid] = gained
            if pid == NO_PURCHASE_ID:
                net_new += max(gained, 0.0)
            elif design.nest_of_product[pid] == focal_nest:
                within_brand += max(gained, 0.0)
            else:
                competitor += max(gained, 0.0)
        segments.append(
            SegmentRedistribution(
                segment_id=seg_id,
                focal_units=seg_focal_units,
                within_brand_units=within_brand,
                competitor_units=competitor,
                net_new_units=net_new,
                incumbent_units=seg_incumbent_units,
            )
        )

    redistributed_total = sum(v for v in incumbent_units.values())
    mass_balance = MassBalanceCheck(
        focal_units_removed=total_focal_units,
        redistributed_units_total=redistributed_total,
        tolerance=mass_balance_tolerance,
    )

    return CounterfactualOutcome(
        focal_product_id=focal_product_id,
        total_focal_units=total_focal_units,
        incumbent_units=incumbent_units,
        segments=tuple(segments),
        mass_balance=mass_balance,
    )
