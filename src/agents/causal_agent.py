"""Causal agent (build plan section 8, node 5): runs the method `method_selector`
chose, plus its cross-check(s), via the tools layer only. This node's entire job is
dispatch -- every number in the returned `CannibalizationResult` traces back to a
`src/causal/` function call, never to LLM output (see `validator.py` for the assertion
that enforces this at the graph level).

Two data regimes, matching `method_rules.yaml`:
  * Structural (`mixed_logit`, `nested_logit_individual`, `blp_nested_logit`) -- fit a
    nested-logit choice model (aggregate grouped-MLE by default; see `src/tools.py`
    `build_aggregate_design`) and read the cannibalization rate off its counterfactual.
  * Reduced-form (`synthetic_control`, `causal_impact`, `did`) -- per incumbent, treat
    the *other* incumbents in the choice set as the donor/control pool (the same
    products `data_agent._classify_donor_pool` already counted), forecast/synthesize
    each incumbent's counterfactual, and sum the implied cannibalized units. This is a
    v1 simplification -- a production deployment would pass an explicit donor list of
    products verified unaffected by the focal launch (see build plan section 13).
  * `bayesian_shrinkage_logit` -- the data-poor fallback: a substitutability-weighted
    heuristic split with deliberately wide, low-confidence uncertainty. Never treated
    as more than directional.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from src.causal.causal_impact import run_causal_impact
from src.causal.did import run_did
from src.causal.nested_logit import FittedChoiceModel
from src.causal.synthetic_control import run_synthetic_control
from src.causal.uncertainty import bootstrap_ci
from src.schema.core import Product, SalesObservation
from src.schema.results import CannibalizationResult, MethodName, ReducedFormResult
from src.schema.tools import ChoiceSet, MethodSelection
from src.tools import (
    ReducedFormSeries,
    build_aggregate_design,
    build_reduced_form_series,
    fit_structural_model,
)

STRUCTURAL_METHODS: frozenset[MethodName] = frozenset(
    {"mixed_logit", "nested_logit_individual", "blp_nested_logit"}
)
REDUCED_FORM_METHODS: frozenset[MethodName] = frozenset(
    {"synthetic_control", "causal_impact", "did"}
)


class CausalAgentResult:
    """Everything the validator/reporter need, bundled -- not a Pydantic model because
    it carries the mutable `FittedChoiceModel` (only present for structural methods).
    """

    def __init__(
        self,
        preliminary_result: CannibalizationResult,
        fitted_model: FittedChoiceModel | None,
        primary_reduced_form: list[ReducedFormResult],
        cross_check_reduced_form: list[ReducedFormResult],
        cross_check_overall_rate: float | None = None,
    ) -> None:
        self.preliminary_result = preliminary_result
        self.fitted_model = fitted_model
        self.primary_reduced_form = primary_reduced_form
        self.cross_check_reduced_form = cross_check_reduced_form
        # normalized the same way as preliminary_result.overall_rate (fraction of
        # total_focal_units), so validator.py can compare them directly
        self.cross_check_overall_rate = cross_check_overall_rate


def run_causal_analysis(
    products: list[Product],
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
    method_selection: MethodSelection,
    seed: int = 42,
) -> CausalAgentResult:
    if method_selection.primary_method in STRUCTURAL_METHODS:
        return _run_structural(products, sales, choice_set, method_selection, seed)
    if method_selection.primary_method in REDUCED_FORM_METHODS:
        return _run_reduced_form(products, sales, choice_set, method_selection, seed)
    return _run_bayesian_shrinkage(products, sales, choice_set, method_selection, seed)


def _run_structural(
    products: list[Product],
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
    method_selection: MethodSelection,
    seed: int,
) -> CausalAgentResult:
    from src.causal.counterfactual import compute_counterfactual

    design = build_aggregate_design(sales, choice_set)
    fitted = fit_structural_model(design, engine="native_mle")
    outcome = compute_counterfactual(fitted, choice_set.focal_product_id)
    incumbent_ids = choice_set.candidate_incumbent_ids

    ci_report = bootstrap_ci(
        fitted, choice_set.focal_product_id, incumbent_ids, n_resamples=200, seed=seed
    )
    segments = outcome.segment_results(incumbent_ids)

    result = CannibalizationResult(
        focal_product_id=choice_set.focal_product_id,
        overall_rate=ci_report.point_estimate,
        ci_low=ci_report.ci_low,
        ci_high=ci_report.ci_high,
        segments=segments,
        method_used=method_selection.primary_method,
        cross_check_methods=method_selection.cross_check_methods,
        confidence="high",  # provisional -- validator.py has the final say
        seed=seed,
        mass_balance=outcome.mass_balance,
    )
    return CausalAgentResult(result, fitted, [], [])


def _incumbent_launch_date(products: list[Product]) -> date:
    focal = next((p for p in products if p.is_focal), None)
    if focal is None or focal.launch_date is None:
        raise ValueError("reduced-form methods require the focal product's launch_date")
    return focal.launch_date


def _run_one_reduced_form(
    method: MethodName, series: ReducedFormSeries, incumbent_id: str
) -> ReducedFormResult:
    if method == "synthetic_control":
        return run_synthetic_control(
            incumbent_id,
            series.incumbent_pre,
            series.donor_pre,
            series.incumbent_post,
            series.donor_post,
            donor_ids=series.donor_ids,
        )
    if method == "causal_impact":
        return run_causal_impact(
            incumbent_id,
            series.incumbent_pre,
            series.donor_pre,
            series.incumbent_post,
            series.donor_post,
        )
    if method == "did":
        pre_len, post_len = len(series.periods_pre), len(series.periods_post)
        incumbent_full = np.concatenate([series.incumbent_pre, series.incumbent_post])
        donor_rows = [
            np.concatenate([series.donor_pre[:, d], series.donor_post[:, d]])
            for d in range(len(series.donor_ids))
        ]
        units = np.vstack([incumbent_full, *donor_rows]) if donor_rows else incumbent_full[None, :]
        post_mask = np.array([False] * pre_len + [True] * post_len)
        result, _passed = run_did(
            incumbent_id, [incumbent_id, *series.donor_ids], 0, units, post_mask
        )
        return result
    raise ValueError(f"unsupported reduced-form method: {method}")


def _aggregate_reduced_form(
    products: list[Product],
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
    method: MethodName,
    launch_date: date,
) -> tuple[list[ReducedFormResult], float, float, float]:
    """Runs `method` per incumbent (donors = the other incumbents) and sums the
    implied cannibalized units. Returns (per-incumbent results, total cannibalized
    units, ci_low, ci_high). A negative `impact_units` (actual < counterfactual) means
    the incumbent lost units relative to its counterfactual -- i.e. was cannibalized.
    """
    incumbent_ids = choice_set.candidate_incumbent_ids
    results: list[ReducedFormResult] = []
    total, ci_low_total, ci_high_total = 0.0, 0.0, 0.0
    for incumbent_id in incumbent_ids:
        donor_ids = [pid for pid in incumbent_ids if pid != incumbent_id]
        if not donor_ids:
            continue
        series = build_reduced_form_series(sales, incumbent_id, donor_ids, launch_date)
        if len(series.periods_pre) < 2 or len(series.periods_post) < 1:
            continue
        result = _run_one_reduced_form(method, series, incumbent_id)
        results.append(result)
        cannibalized = max(-result.impact_units, 0.0)
        total += cannibalized
        ci_low_total += -result.ci_high
        ci_high_total += -result.ci_low
    return results, total, max(ci_low_total, 0.0), max(ci_high_total, 0.0)


def _run_reduced_form(
    products: list[Product],
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
    method_selection: MethodSelection,
    seed: int,
) -> CausalAgentResult:
    launch_date = _incumbent_launch_date(products)
    focal_id = choice_set.focal_product_id
    total_focal_units = sum(
        obs.units for obs in sales if obs.product_id == focal_id and obs.period >= launch_date
    )

    primary_results, cannibalized, ci_low, ci_high = _aggregate_reduced_form(
        products, sales, choice_set, method_selection.primary_method, launch_date
    )
    overall_rate = cannibalized / total_focal_units if total_focal_units > 0 else 0.0
    rate_ci_low = ci_low / total_focal_units if total_focal_units > 0 else 0.0
    rate_ci_high = ci_high / total_focal_units if total_focal_units > 0 else 0.0

    cross_check_results: list[ReducedFormResult] = []
    cross_check_overall_rate: float | None = None
    if method_selection.cross_check_methods:
        cross_check_results, cc_cannibalized, _, _ = _aggregate_reduced_form(
            products, sales, choice_set, method_selection.cross_check_methods[0], launch_date
        )
        if total_focal_units > 0:
            cross_check_overall_rate = cc_cannibalized / total_focal_units

    result = CannibalizationResult(
        focal_product_id=focal_id,
        overall_rate=overall_rate,
        ci_low=min(rate_ci_low, overall_rate),
        ci_high=max(rate_ci_high, overall_rate),
        segments=[],
        method_used=method_selection.primary_method,
        cross_check_methods=method_selection.cross_check_methods,
        confidence="high",  # provisional -- validator.py has the final say
        seed=seed,
    )
    return CausalAgentResult(
        result, None, primary_results, cross_check_results, cross_check_overall_rate
    )


def _run_bayesian_shrinkage(
    products: list[Product],
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
    method_selection: MethodSelection,
    seed: int,
) -> CausalAgentResult:
    """Data-poor fallback: split focal's implied demand across incumbents in
    proportion to substitutability score (a weak prior, not an estimate), with a wide
    +/-50%-relative uncertainty band. `validator.py` enforces the confidence ceiling
    this rule sets in `method_rules.yaml` regardless of anything computed here.
    """
    incumbent_scores = {
        e.product_id: e.substitutability_to_focal
        for e in choice_set.entries
        if e.product_id in choice_set.candidate_incumbent_ids
    }
    total_score = sum(incumbent_scores.values()) or 1.0
    # a directional prior: assume half of focal's demand is net-new, the rest split
    # across incumbents by substitutability -- explicitly a heuristic, not a fit
    prior_overall_rate = 0.5 * sum(incumbent_scores.values()) / (total_score or 1.0)
    prior_overall_rate = float(np.clip(prior_overall_rate, 0.0, 1.0))

    result = CannibalizationResult(
        focal_product_id=choice_set.focal_product_id,
        overall_rate=prior_overall_rate,
        ci_low=max(prior_overall_rate - 0.5, 0.0),
        ci_high=min(prior_overall_rate + 0.5, 1.0),
        segments=[],
        method_used="bayesian_shrinkage_logit",
        cross_check_methods=[],
        confidence="low",
        seed=seed,
        narrative="Insufficient data for a fitted estimate; this is a directional, "
        "substitutability-weighted prior with deliberately wide uncertainty.",
    )
    return CausalAgentResult(result, None, [], [])
