from datetime import date

import numpy as np

from src.schema.core import Product, SalesObservation
from src.signals.segmentation import region_or_price_tier_segments, segment_customers


def test_segment_customers_clusters_by_loyalty_and_price_sensitivity() -> None:
    rng = np.random.default_rng(0)
    n = 200
    # two well-separated synthetic clusters
    loyal_high = rng.normal(0.9, 0.03, n // 2)
    loyal_low = rng.normal(0.1, 0.03, n // 2)
    loyalty = np.concatenate([loyal_high, loyal_low])
    price_sensitivity = np.concatenate(
        [rng.normal(0.2, 0.03, n // 2), rng.normal(0.8, 0.03, n // 2)]
    )
    income = rng.normal(50000, 1000, n)
    sizes = np.ones(n)

    result = segment_customers(loyalty, price_sensitivity, income, sizes, n_segments=2)
    assert result.method == "clustering"
    assert len(result.segments) == 2
    loyalties = sorted(s.loyalty_score for s in result.segments)
    assert loyalties[0] < 0.3
    assert loyalties[1] > 0.7


def test_region_or_price_tier_fallback_segmentation() -> None:
    products = [
        Product(
            product_id="A1", name="A1", category="c", brand="b", price_tier="premium", is_focal=True
        ),
        Product(product_id="A2", name="A2", category="c", brand="b", price_tier="value"),
    ]
    sales = [
        SalesObservation(
            product_id="A1", region="US", period=date(2024, 1, 1), units=100, price=10
        ),
        SalesObservation(product_id="A2", region="US", period=date(2024, 1, 1), units=50, price=5),
        SalesObservation(product_id="A1", region="EU", period=date(2024, 1, 1), units=30, price=10),
    ]
    result = region_or_price_tier_segments(products, sales)
    assert result.method == "region_or_price_tier"
    sizes = {s.segment_id: s.size for s in result.segments}
    assert sizes["US__premium"] == 100
    assert sizes["US__value"] == 50
    assert sizes["EU__premium"] == 30
