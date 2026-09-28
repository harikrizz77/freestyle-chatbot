"""Customer segmentation (build plan section 9). With panel/transaction data, cluster
on loyalty (repeat-purchase/tenure), price sensitivity, and income proxy. Without
panel data, fall back to region / price-tier segments -- never fabricate segments
the data can't support.

`segments_estimable` in `DataProfile` (consumed by `method_selector` via
`method_rules.yaml`) is set from whether this module can actually produce
`CustomerSegment`s, not from a guess.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.schema.core import CustomerSegment, Product, SalesObservation
from src.schema.tools import SegmentationResult


def segment_customers(
    loyalty_scores: np.ndarray,
    price_sensitivities: np.ndarray,
    income_proxies: np.ndarray,
    segment_sizes: np.ndarray,
    n_segments: int = 4,
    seed: int = 42,
) -> SegmentationResult:
    """K-means clustering on (loyalty, price_sensitivity, income_proxy), one row per
    household/customer (or per aggregated panel cell). Requires panel-level inputs --
    callers without panel data should use `region_or_price_tier_segments` instead.
    """
    n = len(loyalty_scores)
    if n < n_segments:
        n_segments = max(1, n)

    features = np.column_stack([loyalty_scores, price_sensitivities, income_proxies])
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=n_segments, random_state=seed, n_init=10).fit_predict(scaled)

    segments = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        segments.append(
            CustomerSegment(
                segment_id=f"cluster_{cluster_id}",
                loyalty_score=float(np.clip(loyalty_scores[mask].mean(), 0, 1)),
                price_sensitivity=float(price_sensitivities[mask].mean()),
                income_proxy=float(income_proxies[mask].mean()),
                size=float(segment_sizes[mask].sum()),
            )
        )
    return SegmentationResult(segments=segments, method="clustering")


def region_or_price_tier_segments(
    products: list[Product], observations: list[SalesObservation]
) -> SegmentationResult:
    """Fallback segmentation when no panel/transaction data exists: one pseudo-segment
    per region x price-tier cell, sized by observed units. Loyalty/income aren't
    estimable at this granularity, so they're left at neutral defaults (0.5) rather
    than fabricated -- `dominant_source` attribution downstream should be read with
    that caveat when `method == "region_or_price_tier"`.
    """
    price_tier_by_product = {p.product_id: p.price_tier for p in products}
    cells: dict[tuple[str, str], float] = {}
    for obs in observations:
        tier = price_tier_by_product.get(obs.product_id, "unknown")
        key = (obs.region, tier)
        cells[key] = cells.get(key, 0.0) + obs.units

    segments = [
        CustomerSegment(
            segment_id=f"{region}__{tier}",
            loyalty_score=0.5,
            price_sensitivity=0.5,
            income_proxy=0.5,
            size=size,
        )
        for (region, tier), size in cells.items()
    ]
    return SegmentationResult(segments=segments, method="region_or_price_tier")
