from datetime import date, timedelta

from src.agents.causal_agent import run_causal_analysis
from src.schema.core import AnalysisRequest, Product, SalesObservation
from src.schema.tools import MethodSelection
from src.tools import build_choice_set

LAUNCH = date(2024, 3, 4)


def _products() -> list[Product]:
    return [
        Product(
            product_id="F1",
            name="Focal",
            category="c",
            brand="FocalBrand",
            price_tier="mid",
            is_focal=True,
            launch_date=LAUNCH,
        ),
        Product(
            product_id="I1", name="Incumbent 1", category="c", brand="RivalBrand", price_tier="mid"
        ),
        Product(
            product_id="I2", name="Incumbent 2", category="c", brand="RivalBrand", price_tier="mid"
        ),
    ]


def _sales() -> list[SalesObservation]:
    sales = []
    for w in range(18):
        period = date(2024, 1, 1) + timedelta(weeks=w)
        pre = period < LAUNCH
        sales.append(
            SalesObservation(
                product_id="I1", region="US", period=period, units=100.0 if pre else 80.0, price=5.0
            )
        )
        sales.append(
            SalesObservation(product_id="I2", region="US", period=period, units=50.0, price=5.0)
        )
        if not pre:
            sales.append(
                SalesObservation(product_id="F1", region="US", period=period, units=30.0, price=6.0)
            )
    return sales


def _choice_set():
    products = _products()
    request = AnalysisRequest(focal_product_id="F1", window=(date(2024, 1, 1), date(2024, 5, 1)))
    return products, request, build_choice_set(products, request)


def test_synthetic_control_reduced_form_path_runs() -> None:
    products, _request, choice_set = _choice_set()
    sales = _sales()
    selection = MethodSelection(
        rule_id="test",
        primary_method="synthetic_control",
        cross_check_methods=["did"],
        segmentation="none",
    )
    result = run_causal_analysis(products, sales, choice_set, selection, seed=1)

    assert result.preliminary_result.method_used == "synthetic_control"
    assert result.preliminary_result.overall_rate > 0
    assert len(result.primary_reduced_form) >= 1
    assert result.cross_check_overall_rate is not None
    assert result.fitted_model is None  # reduced-form path never fits a choice model


def test_did_reduced_form_path_runs() -> None:
    products, _request, choice_set = _choice_set()
    sales = _sales()
    selection = MethodSelection(
        rule_id="test", primary_method="did", cross_check_methods=[], segmentation="none"
    )
    result = run_causal_analysis(products, sales, choice_set, selection, seed=1)

    assert result.preliminary_result.method_used == "did"
    assert result.preliminary_result.overall_rate >= 0
    assert result.cross_check_overall_rate is None


def test_bayesian_shrinkage_fallback_is_low_confidence() -> None:
    products, _request, choice_set = _choice_set()
    selection = MethodSelection(
        rule_id="sparse_bayesian_fallback",
        primary_method="bayesian_shrinkage_logit",
        cross_check_methods=[],
        segmentation="none",
        confidence_ceiling="low",
    )
    result = run_causal_analysis(products, [], choice_set, selection, seed=1)

    assert result.preliminary_result.method_used == "bayesian_shrinkage_logit"
    assert result.preliminary_result.confidence == "low"
    assert 0.0 <= result.preliminary_result.overall_rate <= 1.0
    assert result.preliminary_result.ci_high - result.preliminary_result.ci_low >= 0.5
