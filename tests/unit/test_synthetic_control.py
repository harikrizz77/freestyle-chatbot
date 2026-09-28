import numpy as np
import pytest

from src.causal.synthetic_control import run_synthetic_control


def test_recovers_known_donor_weights_and_impact() -> None:
    rng = np.random.default_rng(0)
    t_pre, t_post = 20, 10
    donor1_pre = rng.uniform(50, 150, size=t_pre)
    donor2_pre = rng.uniform(50, 150, size=t_pre)
    true_weights = np.array([0.7, 0.3])
    incumbent_pre = true_weights[0] * donor1_pre + true_weights[1] * donor2_pre

    donor1_post = rng.uniform(50, 150, size=t_post)
    donor2_post = rng.uniform(50, 150, size=t_post)
    counterfactual_post = true_weights[0] * donor1_post + true_weights[1] * donor2_post
    true_impact_per_period = -8.0
    incumbent_post = counterfactual_post + true_impact_per_period

    result = run_synthetic_control(
        incumbent_product_id="incumbent_1",
        pre_incumbent=incumbent_pre,
        pre_donors=np.column_stack([donor1_pre, donor2_pre]),
        post_incumbent=incumbent_post,
        post_donors=np.column_stack([donor1_post, donor2_post]),
        donor_ids=["donor1", "donor2"],
    )

    assert result.pre_period_fit_score == pytest.approx(1.0, abs=1e-6)
    assert result.impact_units == pytest.approx(true_impact_per_period * t_post, rel=1e-3)
    assert result.ci_low <= result.impact_units <= result.ci_high
    assert result.diagnostics["weight_donor1"] == pytest.approx(0.7, abs=0.05)
    assert result.diagnostics["weight_donor2"] == pytest.approx(0.3, abs=0.05)


def test_empty_donor_pool_raises() -> None:
    with pytest.raises(ValueError):
        run_synthetic_control(
            incumbent_product_id="incumbent_1",
            pre_incumbent=np.ones(5),
            pre_donors=np.zeros((5, 0)),
            post_incumbent=np.ones(3),
            post_donors=np.zeros((3, 0)),
        )
