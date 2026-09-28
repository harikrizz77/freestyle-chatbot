"""Phase 1 acceptance: the native MLE nested-logit estimator recovers known generating
parameters from simulated choice data, and reports a log-likelihood better than the
equal-shares baseline (a lightweight stand-in for the holdout hit-rate check, since
real licensed Dominick's/Nielsen data isn't fetchable in this sandbox -- see the
fixture's module docstring).
"""

import numpy as np
import pytest

from src.causal.nested_logit import fit_nested_logit
from tests.fixtures.simulated_transactions import (
    COMPETITOR_ID,
    FOCAL_ID,
    INCUMBENT_ID,
    TRUE_COEFFICIENTS,
    TRUE_LAMBDAS,
    simulate_aggregate_design,
    simulate_transaction_design,
)


def test_mle_recovers_beta_signs_and_magnitude() -> None:
    design = simulate_transaction_design(n=2500, seed=1)
    fitted = fit_nested_logit(design)

    assert fitted.coefficients.beta_loyalty > 0
    assert fitted.coefficients.beta_price < 0
    assert fitted.coefficients.beta_loyalty == pytest.approx(
        TRUE_COEFFICIENTS.beta_loyalty, rel=0.35
    )
    assert fitted.coefficients.beta_price == pytest.approx(TRUE_COEFFICIENTS.beta_price, rel=0.35)
    for pid in (FOCAL_ID, INCUMBENT_ID, COMPETITOR_ID):
        assert fitted.coefficients.intercepts[pid] == pytest.approx(
            TRUE_COEFFICIENTS.intercepts[pid], abs=0.35
        )


def test_mle_recovers_nesting_parameter() -> None:
    design = simulate_transaction_design(n=2500, seed=1)
    fitted = fit_nested_logit(design)
    assert 0.0 < fitted.lambdas["BrandA"] < 1.0
    assert fitted.lambdas["BrandA"] == pytest.approx(TRUE_LAMBDAS["BrandA"], abs=0.25)


def test_log_likelihood_beats_equal_shares_baseline() -> None:
    design = simulate_transaction_design(n=2500, seed=1)
    fitted = fit_nested_logit(design)
    n_products = design.n_products
    baseline_ll = design.n_obs * np.log(1.0 / n_products)
    assert fitted.log_likelihood is not None
    assert fitted.log_likelihood > baseline_ll


def test_mle_recovers_params_from_aggregate_choice_weights() -> None:
    """The grouped/weighted likelihood path (aggregate scanner/shipment data with
    observed unit counts per market, no per-row chosen alternative) recovers the same
    generating parameters as the individual-level path."""
    design = simulate_aggregate_design(n_markets=500, units_per_market=200, seed=1)
    fitted = fit_nested_logit(design)

    assert fitted.coefficients.beta_loyalty > 0
    assert fitted.coefficients.beta_price < 0
    assert fitted.coefficients.beta_loyalty == pytest.approx(
        TRUE_COEFFICIENTS.beta_loyalty, rel=0.35
    )
    assert fitted.coefficients.beta_price == pytest.approx(TRUE_COEFFICIENTS.beta_price, rel=0.35)
    assert 0.0 < fitted.lambdas["BrandA"] < 1.0


def test_fitted_model_has_coefficient_covariance_for_uncertainty() -> None:
    design = simulate_transaction_design(n=1500, seed=2)
    fitted = fit_nested_logit(design)
    assert fitted.coefficient_cov is not None
    assert fitted.param_layout is not None
    n_params = fitted.param_layout.n_params
    assert fitted.coefficient_cov.shape == (n_params, n_params)
