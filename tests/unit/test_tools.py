from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.causal.counterfactual import compute_counterfactual
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.schema.core import AnalysisRequest
from src.tools import build_aggregate_design, build_choice_set, build_reduced_form_series

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
