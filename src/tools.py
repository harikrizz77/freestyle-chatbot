"""The typed tools layer (build plan section 7): deterministic, independently
unit-tested functions wrapping the causal core, data adapters, and signals layer.
Every function takes/returns a Pydantic model from `src/schema/`. LLM agents in
`src/agents/` call these -- they never compute a number themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from src.causal.nested_logit import (
    NO_PURCHASE_ID,
    OUTSIDE_NEST,
    Engine,
    FittedChoiceModel,
    NestedLogitDesign,
    fit_nested_logit,
    fit_via_pyblp,
    fit_via_pylogit,
    fit_via_xlogit,
)
from src.data.base_adapter import DataAdapter
from src.schema.core import AnalysisRequest, Product, SalesObservation
from src.schema.tools import ChoiceSet, ChoiceSetEntry

_ENGINE_DISPATCH = {
    "pyblp": fit_via_pyblp,
    "pylogit": fit_via_pylogit,
    "xlogit": fit_via_xlogit,
}


def load_data(adapter: DataAdapter, request: AnalysisRequest) -> list[SalesObservation]:
    """`source` is any `DataAdapter` instance -- industry specifics live entirely in
    the adapter; this function never branches on source type."""
    return adapter.load_sales(request)


def _numeric_attribute_vector(product: Product, keys: list[str]) -> np.ndarray:
    return np.array(
        [
            float(product.attributes[k]) if _is_float(product.attributes.get(k)) else 0.0
            for k in keys
        ]
    )


def _is_float(value: object) -> bool:
    if value is None:
        return False
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def build_choice_set(
    products: list[Product], request: AnalysisRequest, min_substitutability: float = 0.0
) -> ChoiceSet:
    """Domain-agnostic substitutability via attribute-embedding cosine similarity, plus
    brand/price-tier bonuses -- no industry-specific branching, per build plan 7.3.
    Always includes the no-purchase outside option.
    """
    focal = next(
        (p for p in products if p.is_focal or p.product_id == request.focal_product_id), None
    )
    if focal is None:
        raise ValueError(f"focal product {request.focal_product_id!r} not found in products")

    candidates = [p for p in products if p.product_id != focal.product_id]
    if request.candidate_incumbents is not None:
        wanted = set(request.candidate_incumbents)
        candidates = [p for p in candidates if p.product_id in wanted]

    numeric_keys = sorted(
        {k for p in [focal, *candidates] for k, v in p.attributes.items() if _is_float(v)}
    )
    focal_vec = _numeric_attribute_vector(focal, numeric_keys)
    focal_norm = float(np.linalg.norm(focal_vec)) or 1.0

    entries = [
        ChoiceSetEntry(
            product_id=focal.product_id,
            nest=focal.nest or focal.brand,
            substitutability_to_focal=1.0,
            is_focal=True,
        )
    ]
    kept_incumbents: list[str] = []
    for cand in candidates:
        cand_vec = _numeric_attribute_vector(cand, numeric_keys)
        cand_norm = float(np.linalg.norm(cand_vec)) or 1.0
        cosine = (
            float(np.dot(focal_vec, cand_vec) / (focal_norm * cand_norm)) if numeric_keys else 0.0
        )
        cosine = (cosine + 1) / 2  # map [-1, 1] -> [0, 1]
        brand_bonus = 0.3 if cand.brand == focal.brand else 0.0
        tier_bonus = 0.15 if cand.price_tier == focal.price_tier else 0.0
        score = float(np.clip(0.55 * cosine + brand_bonus + tier_bonus, 0.0, 1.0))
        if score < min_substitutability:
            continue
        entries.append(
            ChoiceSetEntry(
                product_id=cand.product_id,
                nest=cand.nest or cand.brand,
                substitutability_to_focal=score,
                is_focal=False,
            )
        )
        kept_incumbents.append(cand.product_id)

    entries.append(
        ChoiceSetEntry(
            product_id=NO_PURCHASE_ID,
            nest=OUTSIDE_NEST,
            substitutability_to_focal=0.0,
            is_focal=False,
        )
    )

    return ChoiceSet(
        focal_product_id=focal.product_id,
        entries=entries,
        candidate_incumbent_ids=kept_incumbents,
    )


def build_aggregate_design(
    sales: list[SalesObservation],
    choice_set: ChoiceSet,
) -> NestedLogitDesign:
    """Turns aggregate region x period sales (the shape scanner/registrations/shipment
    adapters actually provide) into a `NestedLogitDesign` with one row per market and
    `choice_weights` = observed unit counts, rather than a per-row chosen alternative.
    `fit_nested_logit` treats this as the grouped-multinomial-logit MLE -- the same
    model family individual-level data fits, just with counts instead of single draws.

    Requires every `SalesObservation.market_size` to be populated (the adapter's
    market-size proxy); this is what lets the no-purchase share be inferred as the gap
    between total market size and observed unit sales.
    """
    product_order = [e.product_id for e in choice_set.entries]
    if product_order[-1] != NO_PURCHASE_ID:
        raise ValueError("choice_set.entries must end with the no-purchase option")
    nest_of_product = {e.product_id: e.nest for e in choice_set.entries}

    markets: dict[tuple[str, date], dict[str, tuple[float, float, float]]] = {}
    for obs in sales:
        if obs.market_size is None:
            raise ValueError(
                f"build_aggregate_design requires market_size on every SalesObservation "
                f"(missing for product={obs.product_id}, region={obs.region}, period={obs.period})"
            )
        key = (obs.region, obs.period)
        markets.setdefault(key, {})[obs.product_id] = (obs.units, obs.price, obs.market_size)

    market_keys = sorted(markets.keys())
    n = len(market_keys)
    j = len(product_order)
    if n == 0:
        raise ValueError("no sales observations to build an aggregate design from")

    market_size = np.zeros(n)
    price_over_income = np.zeros((n, j))
    choice_weights = np.zeros((n, j))
    segment_ids = [f"{region}__{period.isoformat()}" for region, period in market_keys]

    for i, key in enumerate(market_keys):
        row = markets[key]
        any_market_size = next(iter(row.values()))[2]
        market_size[i] = any_market_size
        real_units_total = 0.0
        for j_idx, pid in enumerate(product_order[:-1]):  # exclude no-purchase
            if pid in row:
                units, price, _ = row[pid]
                price_over_income[i, j_idx] = price
                choice_weights[i, j_idx] = units
                real_units_total += units
        choice_weights[i, -1] = max(any_market_size - real_units_total, 0.0)

    zeros = np.zeros((n, j))
    design = NestedLogitDesign(
        product_ids=tuple(product_order),
        nest_of_product=nest_of_product,
        segment_ids=tuple(segment_ids),
        market_size=market_size,
        loyalty=zeros,
        price_over_income=price_over_income,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
        choice_weights=choice_weights,
    )
    return design


@dataclass(frozen=True)
class ReducedFormSeries:
    """Pre/post period unit series for one incumbent and its donor pool, aligned on a
    common sorted period axis and split at the focal launch date -- the input shape
    `run_synthetic_control` / `run_causal_impact` / `run_did` all consume."""

    periods_pre: list[date]
    periods_post: list[date]
    incumbent_pre: np.ndarray
    incumbent_post: np.ndarray
    donor_pre: np.ndarray  # (T_pre, D)
    donor_post: np.ndarray  # (T_post, D)
    donor_ids: list[str]


def build_reduced_form_series(
    sales: list[SalesObservation], incumbent_id: str, donor_ids: list[str], launch_date: date
) -> ReducedFormSeries:
    """Aggregates units across regions for `incumbent_id` and each donor, per period,
    then splits at `launch_date`. Periods missing for any series are dropped (only
    common periods are kept) so the resulting arrays are always aligned.
    """
    by_product_period: dict[str, dict[date, float]] = {}
    for obs in sales:
        if obs.product_id not in {incumbent_id, *donor_ids}:
            continue
        by_product_period.setdefault(obs.product_id, {})
        by_product_period[obs.product_id][obs.period] = (
            by_product_period[obs.product_id].get(obs.period, 0.0) + obs.units
        )

    if incumbent_id not in by_product_period:
        raise ValueError(f"no sales found for incumbent {incumbent_id!r}")
    missing_donors = [d for d in donor_ids if d not in by_product_period]
    if missing_donors:
        raise ValueError(f"no sales found for donor(s): {missing_donors}")

    common_periods = sorted(
        set(by_product_period[incumbent_id]).intersection(
            *(set(by_product_period[d]) for d in donor_ids)
        )
    )
    periods_pre = [p for p in common_periods if p < launch_date]
    periods_post = [p for p in common_periods if p >= launch_date]

    def series_for(pid: str, periods: list[date]) -> np.ndarray:
        return np.array([by_product_period[pid][p] for p in periods])

    return ReducedFormSeries(
        periods_pre=periods_pre,
        periods_post=periods_post,
        incumbent_pre=series_for(incumbent_id, periods_pre),
        incumbent_post=series_for(incumbent_id, periods_post),
        donor_pre=np.column_stack([series_for(d, periods_pre) for d in donor_ids])
        if donor_ids
        else np.zeros((len(periods_pre), 0)),
        donor_post=np.column_stack([series_for(d, periods_post) for d in donor_ids])
        if donor_ids
        else np.zeros((len(periods_post), 0)),
        donor_ids=donor_ids,
    )


def fit_structural_model(
    design: NestedLogitDesign, engine: Engine = "native_mle"
) -> FittedChoiceModel:
    """Dispatches to the requested structural engine. Falls back to the native
    maximum-likelihood estimator (statistically identical nested-logit model) if the
    requested optional engine isn't installed or its adapter wiring isn't available --
    logged, never silent-guessed as a *different* model.
    """
    if engine == "native_mle":
        return fit_nested_logit(design)

    import structlog

    log = structlog.get_logger(__name__)
    try:
        return _ENGINE_DISPATCH[engine](design)
    except (ImportError, NotImplementedError) as exc:
        log.warning(
            "structural_engine_fallback",
            requested_engine=engine,
            reason=str(exc),
            fallback="native_mle",
        )
        return fit_nested_logit(design)
