from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.causal.counterfactual import compute_counterfactual
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.schema.core import AnalysisRequest
from src.schema.signals import CompetitorEvent, MacroSeries
from src.tools import (
    build_aggregate_design,
    build_choice_set,
    build_market_context,
    build_reduced_form_series,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def cpg_products_and_sales():
    adapter = CPGScannerAdapter(FIXTURES / "cpg_scanner_sample.csv")
    request = AnalysisRequest(
        focal_product_id="1002", window=(date(2024, 1, 1), date(2024, 12, 31))
    )
    return adapter.load_products(request), adapter.load_sales(request), request


def test_build_aggregate_design_end_to_end(cpg_products_and_sales) -> None:
    products, sales, request = cpg_products_and_sales
    choice_set = build_choice_set(products, request)
    design = build_aggregate_design(sales, choice_set)

    assert design.n_obs == 2  # two weeks observed in the fixture
    assert design.product_ids[-1] == "__no_purchase__"
    # no-purchase weight = market_size - observed real units, never negative
    assert np.all(design.choice_weights[:, -1] >= 0)
    # market size covers all observed unit sales for every market row
    assert np.all(design.choice_weights.sum(axis=1) == pytest.approx(design.market_size))

    from src.tools import fit_structural_model

    fitted = fit_structural_model(design)
    outcome = compute_counterfactual(fitted, "1002")
    assert outcome.mass_balance.passed


def test_build_aggregate_design_requires_market_size() -> None:
    from src.schema.core import SalesObservation
    from src.schema.tools import ChoiceSet, ChoiceSetEntry

    sales = [
        SalesObservation(product_id="A1", region="r1", period=date(2024, 1, 1), units=10, price=1.0)
    ]
    choice_set = ChoiceSet(
        focal_product_id="A1",
        entries=[
            ChoiceSetEntry(
                product_id="A1", nest="BrandA", substitutability_to_focal=1.0, is_focal=True
            ),
            ChoiceSetEntry(
                product_id="__no_purchase__", nest="__outside__", substitutability_to_focal=0.0
            ),
        ],
        candidate_incumbent_ids=[],
    )
    with pytest.raises(ValueError, match="market_size"):
        build_aggregate_design(sales, choice_set)


def test_build_reduced_form_series_aligns_and_splits(cpg_products_and_sales) -> None:
    _, sales, _ = cpg_products_and_sales
    series = build_reduced_form_series(
        sales, incumbent_id="1003", donor_ids=["1001"], launch_date=date(2024, 2, 1)
    )
    assert len(series.periods_pre) == 1
    assert len(series.periods_post) == 1
    assert series.incumbent_pre.shape == (1,)
    assert series.donor_pre.shape == (1, 1)
    assert series.incumbent_post[0] == pytest.approx(560.0)


def test_build_reduced_form_series_missing_donor_raises(cpg_products_and_sales) -> None:
    _, sales, _ = cpg_products_and_sales
    with pytest.raises(ValueError, match="no sales found"):
        build_reduced_form_series(
            sales, incumbent_id="1003", donor_ids=["does_not_exist"], launch_date=date(2024, 2, 1)
        )


def test_build_market_context_joins_events_and_macro_as_of_period(cpg_products_and_sales) -> None:
    _, sales, _ = cpg_products_and_sales
    events = [
        CompetitorEvent(
            event_id="e1",
            competitor_brand="RivalCo",
            category="salty_snacks",
            region="store_1",
            event_date=date(2024, 1, 10),  # between the two sales periods
            kind="competitor_launch",
            source="news",
            confidence=0.8,
        )
    ]
    macro = [
        MacroSeries(
            region="store_1",
            indicator="cpi",
            series={date(2024, 1, 7): 300.0, date(2024, 2, 11): 310.0},
            source="test",
        )
    ]
    contexts = build_market_context(sales, events, macro)
    by_period = {c.period: c for c in contexts}

    assert set(by_period) == {date(2024, 1, 7), date(2024, 2, 11)}
    # the event (2024-01-10) is "as-of" joined to the latest period at or before it
    assert by_period[date(2024, 1, 7)].competitor_pressure["RivalCo"] == pytest.approx(0.8)
    assert by_period[date(2024, 1, 7)].macro_index["cpi"] == pytest.approx(300.0)
    assert by_period[date(2024, 2, 11)].competitor_pressure == {}
    assert by_period[date(2024, 2, 11)].macro_index["cpi"] == pytest.approx(310.0)


def test_build_market_context_with_no_signals_returns_empty_contexts(
    cpg_products_and_sales,
) -> None:
    _, sales, _ = cpg_products_and_sales
    contexts = build_market_context(sales)
    assert len(contexts) == 2
    assert all(c.macro_index == {} and c.competitor_pressure == {} for c in contexts)


def test_build_aggregate_design_folds_in_competitor_pressure(cpg_products_and_sales) -> None:
    products, sales, request = cpg_products_and_sales
    choice_set = build_choice_set(products, request)
    events = [
        CompetitorEvent(
            event_id="e1",
            competitor_brand="RivalCo",
            category="salty_snacks",
            region="store_1",
            event_date=date(2024, 1, 10),
            kind="competitor_launch",
            source="news",
            confidence=1.0,
        )
    ]
    market_context = build_market_context(sales, events, [])
    design = build_aggregate_design(
        sales, choice_set, products=products, market_context=market_context
    )

    rival_idx = design.product_index("1003")  # RivalCo Classic Chips
    focal_idx = design.product_index("1002")  # SnackCo Zesty Chips (focal, different brand)
    week1_idx = design.segment_ids.index("store_1__2024-01-07")
    week2_idx = design.segment_ids.index("store_1__2024-02-11")

    assert design.competitor_pressure[week1_idx, rival_idx] == pytest.approx(1.0)
    # a competitor event against RivalCo's own brand shouldn't leak onto other brands
    assert design.competitor_pressure[week1_idx, focal_idx] == pytest.approx(0.0)
    # the event doesn't overlap the second week
    assert design.competitor_pressure[week2_idx, rival_idx] == pytest.approx(0.0)


def test_build_aggregate_design_without_context_leaves_covariates_at_zero(
    cpg_products_and_sales,
) -> None:
    products, sales, request = cpg_products_and_sales
    choice_set = build_choice_set(products, request)
    design = build_aggregate_design(sales, choice_set, products=products)
    assert np.all(design.competitor_pressure == 0.0)
    assert np.all(design.macro_index == 0.0)
