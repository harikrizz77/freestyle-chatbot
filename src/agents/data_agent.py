"""Data agent (build plan section 8, node 2): pick an adapter, load, map to the
normalized schema, and report data granularity/coverage as a `DataProfile` --
the deterministic fact sheet `method_selector` reads via `method_rules.yaml`.
No LLM call is required for this node's core logic; the only place an LLM could
plausibly help is disambiguating a messy adapter choice from a vague request, which
is out of scope for v1 (adapters are curated and selected explicitly, per build plan
section 13's non-goals).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from src.data.base_adapter import DataAdapter
from src.schema.core import AnalysisRequest, Product, SalesObservation
from src.schema.tools import DataLevel, DataProfile, DonorPoolStrength, PriceVariation

_PRICE_CV_SUFFICIENT = 0.05
_PRICE_CV_LOW = 0.01
_DONOR_STRONG_COUNT = 5
_DONOR_WEAK_COUNT = 2


def load_and_profile(
    adapter: DataAdapter, request: AnalysisRequest
) -> tuple[list[Product], list[SalesObservation], DataProfile]:
    """Runs the adapter and computes the data profile in one pass -- the two are
    always consumed together in `graph.py`.
    """
    products = adapter.load_products(request)
    sales = adapter.load_sales(request)
    profile = compute_data_profile(products, sales, request, adapter.data_level())
    return products, sales, profile


def compute_data_profile(
    products: list[Product],
    sales: list[SalesObservation],
    request: AnalysisRequest,
    data_level: DataLevel,
) -> DataProfile:
    focal = next(
        (p for p in products if p.is_focal or p.product_id == request.focal_product_id), None
    )
    price_variation = _classify_price_variation(sales)
    pre_periods = _count_pre_periods(sales, focal)
    donor_pool = _classify_donor_pool(products, focal)
    segments_estimable = data_level in ("transaction", "panel")
    regions = {s.region for s in sales}

    return DataProfile(
        data_level=data_level,
        price_variation=price_variation,
        donor_pool=donor_pool,
        pre_periods=pre_periods,
        segments_estimable=segments_estimable,
        concurrent_shock="none",  # filled in by the competitor/macro signals step
        n_products=len(products),
        n_regions=len(regions),
        coverage_notes=(
            f"{len(sales)} sales rows across {len(products)} products and {len(regions)} regions"
        ),
    )


def _classify_price_variation(sales: list[SalesObservation]) -> PriceVariation:
    if len(sales) < 2:
        return "none"
    prices = np.array([s.price for s in sales])
    mean_price = float(prices.mean())
    if mean_price <= 0:
        return "none"
    cv = float(prices.std()) / mean_price
    if cv >= _PRICE_CV_SUFFICIENT:
        return "sufficient"
    if cv >= _PRICE_CV_LOW:
        return "low"
    return "none"


def _count_pre_periods(sales: list[SalesObservation], focal: Product | None) -> int:
    if focal is None or focal.launch_date is None:
        # no known launch date -- treat the whole window as "pre" is wrong; be
        # conservative and report 0 so method_selector doesn't over-trust a
        # synthetic-control/DiD path that needs a real pre-period.
        return 0
    launch: date = focal.launch_date
    periods = {s.period for s in sales if s.period < launch}
    return len(periods)


def _classify_donor_pool(products: list[Product], focal: Product | None) -> DonorPoolStrength:
    if focal is None:
        return "none"
    same_category = [
        p for p in products if p.category == focal.category and p.product_id != focal.product_id
    ]
    if len(same_category) >= _DONOR_STRONG_COUNT:
        return "strong"
    if len(same_category) >= _DONOR_WEAK_COUNT:
        return "weak"
    return "none"
