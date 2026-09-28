import numpy as np
import pytest

from src.causal.causal_impact import run_causal_impact


def test_recovers_known_impact_with_noiseless_pretrend() -> None:
    rng = np.random.default_rng(0)
    t_pre, t_post = 30, 10
    cov_pre = rng.uniform(0, 1, size=(t_pre, 1))
    beta0, beta1 = 100.0, 20.0
    incumbent_pre = beta0 + beta1 * cov_pre[:, 0]

    cov_post = rng.uniform(0, 1, size=(t_post, 1))
    counterfactual_post = beta0 + beta1 * cov_post[:, 0]
    true_impact_per_period = -5.0
    incumbent_post = counterfactual_post + true_impact_per_period

    result = run_causal_impact(
        incumbent_product_id="incumbent_1",
        pre_incumbent=incumbent_pre,
        pre_covariates=cov_pre,
        post_incumbent=incumbent_post,
        post_covariates=cov_post,
    )

    assert result.pre_period_fit_score == pytest.approx(1.0, abs=1e-6)
    assert result.impact_units == pytest.approx(true_impact_per_period * t_post, rel=1e-3)
    assert result.ci_low <= result.impact_units <= result.ci_high
