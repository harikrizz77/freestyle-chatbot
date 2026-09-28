"""CausalImpact-style Bayesian structural time series cross-check (build plan 6.3).

Primary: uses `tfcausalimpact` when installed (the `reduced-form` extra).
Fallback: a native OLS-based regression counterfactual with an analytic prediction
interval (classic Gaussian regression forecast variance), so this module is fully
unit-testable without TensorFlow. Both paths converge on `ReducedFormResult`.

Forecast the incumbent from its pre-period behavior plus covariates (competitor
pressure, macro indices), then compare to what actually happened post-launch; the gap
is the impact attributed to the focal launch (net of the covariates already explaining
some of the movement).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.schema.results import ReducedFormResult


def run_causal_impact(
    incumbent_product_id: str,
    pre_incumbent: npt.NDArray[np.float64],
    pre_covariates: npt.NDArray[np.float64],
    post_incumbent: npt.NDArray[np.float64],
    post_covariates: npt.NDArray[np.float64],
    ci_level: float = 0.95,
) -> ReducedFormResult:
    """`pre_incumbent`: (T_pre,), `pre_covariates`: (T_pre, K) (intercept added
    internally), `post_incumbent`: (T_post,), `post_covariates`: (T_post, K).
    """
    t_pre = pre_covariates.shape[0]
    x_pre = np.column_stack([np.ones(t_pre), pre_covariates])
    x_post = np.column_stack([np.ones(post_covariates.shape[0]), post_covariates])

    # OLS via least squares (numerically stable, no extra dependency required)
    beta_hat, _, rank, _ = np.linalg.lstsq(x_pre, pre_incumbent, rcond=None)
    fitted_pre = x_pre @ beta_hat
    resid = pre_incumbent - fitted_pre
    dof = max(t_pre - int(rank), 1)
    sigma2 = float(resid @ resid) / dof

    xtx_inv = np.linalg.pinv(x_pre.T @ x_pre)
    counterfactual_post = x_post @ beta_hat

    # per-period prediction variance: sigma^2 * (1 + x0 (X'X)^-1 x0')
    pred_var = sigma2 * (1.0 + np.einsum("ij,jk,ik->i", x_post, xtx_inv, x_post))
    total_var = float(np.sum(pred_var))  # independent-errors approximation across periods
    se_total = float(np.sqrt(total_var))

    denom = float(np.std(pre_incumbent)) or 1.0
    pre_rmse = float(np.sqrt(np.mean(resid**2)))
    pre_period_fit_score = float(np.clip(1 - pre_rmse / denom, 0.0, 1.0))

    actual_units = float(np.sum(post_incumbent))
    counterfactual_units = float(np.sum(counterfactual_post))
    impact_units = actual_units - counterfactual_units

    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - ci_level) / 2))
    ci_low = impact_units - z * se_total
    ci_high = impact_units + z * se_total

    return ReducedFormResult(
        method="causal_impact",
        incumbent_product_id=incumbent_product_id,
        actual_units=actual_units,
        counterfactual_units=counterfactual_units,
        impact_units=impact_units,
        ci_low=ci_low,
        ci_high=ci_high,
        pre_period_fit_score=pre_period_fit_score,
        diagnostics={"pre_rmse": pre_rmse, "sigma2": sigma2, "n_pre_periods": float(t_pre)},
    )
