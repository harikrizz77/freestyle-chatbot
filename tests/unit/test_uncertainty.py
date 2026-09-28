import pytest

from src.causal.nested_logit import fit_nested_logit
from src.causal.uncertainty import bootstrap_ci, cross_check_disagreement, delta_method_ci
from tests.fixtures.simulated_transactions import (
    COMPETITOR_ID,
    FOCAL_ID,
    INCUMBENT_ID,
    simulate_transaction_design,
)


@pytest.fixture(scope="module")
def fitted_model():
    design = simulate_transaction_design(n=1500, seed=3)
    return fit_nested_logit(design)


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic(fitted_model) -> None:
    incumbents = [INCUMBENT_ID, COMPETITOR_ID]
    report_a = bootstrap_ci(fitted_model, FOCAL_ID, incumbents, n_resamples=100, seed=42)
    report_b = bootstrap_ci(fitted_model, FOCAL_ID, incumbents, n_resamples=100, seed=42)

    assert report_a.ci_low <= report_a.point_estimate <= report_a.ci_high
    assert report_a.method == "bootstrap"
    assert report_a.seed == 42
    # same seed -> bit-identical result (reproducibility requirement, build plan 11)
    assert report_a.point_estimate == pytest.approx(report_b.point_estimate)
    assert report_a.ci_low == pytest.approx(report_b.ci_low)
    assert report_a.ci_high == pytest.approx(report_b.ci_high)


def test_bootstrap_ci_different_seed_can_differ(fitted_model) -> None:
    incumbents = [INCUMBENT_ID, COMPETITOR_ID]
    report_a = bootstrap_ci(fitted_model, FOCAL_ID, incumbents, n_resamples=100, seed=1)
    report_b = bootstrap_ci(fitted_model, FOCAL_ID, incumbents, n_resamples=100, seed=2)
    assert report_a.ci_low != report_b.ci_low or report_a.ci_high != report_b.ci_high


def test_delta_method_ci_contains_point_estimate(fitted_model) -> None:
    incumbents = [INCUMBENT_ID, COMPETITOR_ID]
    report = delta_method_ci(fitted_model, FOCAL_ID, incumbents, seed=42)
    assert report.method == "delta_method"
    assert report.ci_low <= report.point_estimate <= report.ci_high


def test_cross_check_disagreement_threshold() -> None:
    assert cross_check_disagreement(0.5, 0.55, focal_units=1000, tolerance=0.10) is False
    assert cross_check_disagreement(0.5, 0.7, focal_units=1000, tolerance=0.10) is True
    assert cross_check_disagreement(0.5, 0.7, focal_units=0, tolerance=0.10) is False
