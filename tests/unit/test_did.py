import numpy as np
import pytest

from src.causal.did import run_did


def _build_panel(
    t_pre: int,
    t_post: int,
    delta_true: float,
    divergent_pretrend: float,
    seed: int,
    noise_sd: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t_total = t_pre + t_post
    entity_fe = np.array([10.0, 5.0])
    time_fe = 0.3 * np.arange(t_total)
    post_mask = np.arange(t_total) >= t_pre

    units = np.zeros((2, t_total))
    for e in range(2):
        units[e, :] = entity_fe[e] + time_fe
    units[0, :] += delta_true * post_mask.astype(float)  # treated = entity 0
    if divergent_pretrend:
        pre_trend = np.arange(t_total, dtype=float)
        pre_trend[post_mask] = pre_trend[t_pre - 1]  # freeze after launch
        units[0, :] += divergent_pretrend * pre_trend
    units += rng.normal(0, noise_sd, size=units.shape)
    return units, post_mask


def test_did_recovers_known_att_with_parallel_pretrends() -> None:
    units, post_mask = _build_panel(
        t_pre=15, t_post=8, delta_true=-4.0, divergent_pretrend=0.0, seed=0
    )
    result, parallel_trends_passed = run_did(
        incumbent_product_id="incumbent_1",
        entity_ids=["incumbent_1", "control_1"],
        treated_entity_index=0,
        units=units,
        post_period_mask=post_mask,
    )
    n_post = int(post_mask.sum())
    assert result.impact_units == pytest.approx(-4.0 * n_post, rel=0.15)
    assert result.ci_low <= result.impact_units <= result.ci_high
    assert parallel_trends_passed is True
    assert result.parallel_trends_passed is True


def test_did_flags_divergent_pretrends() -> None:
    units, post_mask = _build_panel(
        t_pre=15, t_post=8, delta_true=-4.0, divergent_pretrend=0.6, seed=0
    )
    result, parallel_trends_passed = run_did(
        incumbent_product_id="incumbent_1",
        entity_ids=["incumbent_1", "control_1"],
        treated_entity_index=0,
        units=units,
        post_period_mask=post_mask,
    )
    assert parallel_trends_passed is False
    assert result.parallel_trends_passed is False
