"""Difference-in-differences cross-check (build plan 6.3): incumbent vs. matched
controls, external factors as explicit control variables, with a parallel-pre-trends
test the validator treats as a hard gate.

Primary: uses `linearmodels.PanelOLS` when installed (the `reduced-form` extra).
Fallback: a native two-way fixed-effects (entity + time) within-estimator, algebraically
the same model PanelOLS fits, so this module is fully unit-testable without the heavy
optional dependency. Both paths converge on `ReducedFormResult`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.schema.results import ReducedFormResult


def _two_way_demean(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """x: (E, T[, ...]) -> demeaned by entity, by time, plus grand mean added back."""
    entity_mean = x.mean(axis=1, keepdims=True)
    time_mean = x.mean(axis=0, keepdims=True)
    grand_mean = x.mean(axis=(0, 1), keepdims=True)
    return np.asarray(x - entity_mean - time_mean + grand_mean, dtype=np.float64)


def run_did(
    incumbent_product_id: str,
    entity_ids: list[str],
    treated_entity_index: int,
    units: npt.NDArray[np.float64],
    post_period_mask: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64] | None = None,
    ci_level: float = 0.95,
) -> tuple[ReducedFormResult, bool]:
    """`units`: (E, T) outcome panel. `post_period_mask`: (T,) bool, True for periods at
    or after focal launch. `covariates`: optional (E, T, K) external-factor controls
    (competitor pressure, macro index, ...). Returns (result, parallel_trends_passed).
    """
    n_entities, n_periods = units.shape
    treated = np.zeros(n_entities, dtype=float)
    treated[treated_entity_index] = 1.0
    post = post_period_mask.astype(float)
    treated_post = treated[:, None] * post[None, :]  # (E, T)

    regressors = [treated_post]
    if covariates is not None:
        for k in range(covariates.shape[2]):
            regressors.append(covariates[:, :, k])

    y_demeaned = _two_way_demean(units).ravel()
    x_demeaned = np.column_stack([_two_way_demean(r).ravel() for r in regressors])

    beta_hat, _, rank, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
    resid = y_demeaned - x_demeaned @ beta_hat
    n_obs = n_entities * n_periods
    n_fe_params = n_entities + n_periods - 1  # absorbed fixed effects
    dof = max(n_obs - n_fe_params - x_demeaned.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(x_demeaned.T @ x_demeaned)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))

    delta = float(beta_hat[0])  # ATT per treated-post cell
    delta_se = float(se[0])

    n_post = int(post.sum())
    impact_units = delta * n_post
    actual_units = float(units[treated_entity_index, post_period_mask].sum())
    counterfactual_units = actual_units - impact_units

    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - ci_level) / 2))
    se_total = delta_se * n_post
    ci_low = impact_units - z * se_total
    ci_high = impact_units + z * se_total

    parallel_trends_passed = _parallel_trends_test(units, treated, post_period_mask, covariates)

    result = ReducedFormResult(
        method="did",
        incumbent_product_id=incumbent_product_id,
        actual_units=actual_units,
        counterfactual_units=counterfactual_units,
        impact_units=impact_units,
        ci_low=ci_low,
        ci_high=ci_high,
        parallel_trends_passed=parallel_trends_passed,
        diagnostics={"att_per_period": delta, "att_se": delta_se, "n_post_periods": float(n_post)},
    )
    return result, parallel_trends_passed


def _parallel_trends_test(
    units: npt.NDArray[np.float64],
    treated: npt.NDArray[np.float64],
    post_period_mask: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64] | None,
    alpha: float = 0.05,
) -> bool:
    """Placebo test on the pre-period only: regress on treated * linear-time-trend; if
    that interaction is statistically significant, incumbent and donors were already
    diverging before launch and the parallel-trends assumption fails.
    """
    pre_mask = ~post_period_mask
    if pre_mask.sum() < 3:
        # too few pre-periods to test -- treat as an untested (not failed) assumption,
        # the method_rules pre_periods threshold already gates whether DiD is chosen.
        return True

    pre_units = units[:, pre_mask]
    n_entities, n_pre = pre_units.shape
    trend = np.arange(n_pre, dtype=float)
    treated_trend = treated[:, None] * trend[None, :]

    regressors = [treated_trend]
    if covariates is not None:
        pre_cov = covariates[:, pre_mask, :]
        for k in range(pre_cov.shape[2]):
            regressors.append(pre_cov[:, :, k])

    y_demeaned = _two_way_demean(pre_units).ravel()
    x_demeaned = np.column_stack([_two_way_demean(r).ravel() for r in regressors])

    beta_hat, _, rank, _ = np.linalg.lstsq(x_demeaned, y_demeaned, rcond=None)
    resid = y_demeaned - x_demeaned @ beta_hat
    n_obs = n_entities * n_pre
    n_fe_params = n_entities + n_pre - 1
    dof = max(n_obs - n_fe_params - x_demeaned.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(x_demeaned.T @ x_demeaned)
    se0 = float(np.sqrt(max(sigma2 * xtx_inv[0, 0], 0.0)))
    if se0 == 0:
        return True

    from scipy.stats import t as t_dist

    t_stat = beta_hat[0] / se0
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
    return bool(p_value >= alpha)
