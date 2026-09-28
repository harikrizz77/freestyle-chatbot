"""Uncertainty quantification for causal engine outputs.

Two methods, per the build plan:

  * Bootstrap -- resample the fitted coefficient vector from its asymptotic sampling
    distribution (multivariate normal around the MLE, covariance = inverse Hessian of
    the negative log-likelihood), recompute the counterfactual for each draw, and take
    percentiles. This is what `fit_nested_logit` already gives us via
    `FittedChoiceModel.resample_coefficients`.
  * Delta method -- a first-order Taylor expansion of the rate function around theta_hat,
    for when resampling+recomputing is too expensive.

Also home to the mass-balance hard-gate check: removed focal demand must (approximately)
redistribute, not vanish or double-count. A failure here means a bug in the causal core,
not a data-quality issue, and must block output (see validator.py).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.causal.counterfactual import compute_counterfactual
from src.causal.nested_logit import FittedChoiceModel
from src.schema.results import ConfidenceReport, MassBalanceCheck


class MassBalanceError(RuntimeError):
    """Raised when redistributed units don't conserve focal demand. Implementation
    bug, not a data problem -- callers must halt rather than report a number."""


def assert_mass_balance(check: MassBalanceCheck) -> None:
    if not check.passed:
        raise MassBalanceError(
            f"mass balance failed: focal_units_removed={check.focal_units_removed:.6f}, "
            f"redistributed_units_total={check.redistributed_units_total:.6f}, "
            f"discrepancy={check.discrepancy:.6f} (tolerance={check.tolerance})"
        )


def bootstrap_ci(
    fitted: FittedChoiceModel,
    focal_product_id: str,
    incumbent_ids: list[str],
    n_resamples: int = 500,
    seed: int = 42,
    ci_level: float = 0.95,
) -> ConfidenceReport:
    """Resample coefficients, recompute the overall cannibalization rate each time,
    and report the empirical percentile interval. Deterministic given `seed`.
    """
    if fitted.coefficient_cov is None:
        raise ValueError(
            "bootstrap_ci requires fitted.coefficient_cov (from fit_nested_logit); "
            "got a model fitted without an estimated covariance."
        )
    rng = np.random.default_rng(seed)
    point_outcome = compute_counterfactual(fitted, focal_product_id)
    point_estimate = point_outcome.overall_cannibalization_rate(incumbent_ids)

    draws = np.empty(n_resamples)
    for b in range(n_resamples):
        resampled = fitted.resample_coefficients(rng)
        outcome = compute_counterfactual(resampled, focal_product_id)
        draws[b] = outcome.overall_cannibalization_rate(incumbent_ids)

    alpha = 1 - ci_level
    ci_low, ci_high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # keep the interval consistent with the point estimate even under a skewed
    # bootstrap distribution (e.g. all-zero draws from a degenerate fit)
    ci_low = float(min(ci_low, point_estimate))
    ci_high = float(max(ci_high, point_estimate))

    return ConfidenceReport(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        method="bootstrap",
        n_resamples=n_resamples,
        seed=seed,
    )


def delta_method_ci(
    fitted: FittedChoiceModel,
    focal_product_id: str,
    incumbent_ids: list[str],
    seed: int = 42,
    ci_level: float = 0.95,
    step: float = 1e-4,
) -> ConfidenceReport:
    """First-order delta-method CI: Var(rate) ~= grad^T Cov(theta) grad, gradient via
    central finite differences. Cheaper than bootstrap; used as a cross-check or when
    `n_resamples` would be too slow (e.g. very large choice sets).
    """
    if fitted.coefficient_cov is None or fitted.param_layout is None:
        raise ValueError("delta_method_ci requires fitted.coefficient_cov and param_layout")

    layout = fitted.param_layout
    theta_hat = layout.pack(fitted.coefficients, fitted.lambdas)

    def rate_at(theta: npt.NDArray[np.float64]) -> float:
        coefficients, lambdas = layout.unpack(theta)
        model = FittedChoiceModel(
            design=fitted.design,
            coefficients=coefficients,
            lambdas=lambdas,
            engine=fitted.engine,
        )
        return compute_counterfactual(model, focal_product_id).overall_cannibalization_rate(
            incumbent_ids
        )

    point_estimate = rate_at(theta_hat)
    grad = np.empty_like(theta_hat)
    for k in range(len(theta_hat)):
        up, down = theta_hat.copy(), theta_hat.copy()
        up[k] += step
        down[k] -= step
        grad[k] = (rate_at(up) - rate_at(down)) / (2 * step)

    variance = float(grad @ fitted.coefficient_cov @ grad)
    variance = max(variance, 0.0)
    se = float(np.sqrt(variance))

    from scipy.stats import norm

    z = float(norm.ppf(1 - (1 - ci_level) / 2))
    ci_low = point_estimate - z * se
    ci_high = point_estimate + z * se

    return ConfidenceReport(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        method="delta_method",
        n_resamples=None,
        seed=seed,
    )


def cross_check_disagreement(
    rate_a: float, rate_b: float, focal_units: float, tolerance: float
) -> bool:
    """True if two methods' cannibalized-unit estimates diverge by more than
    `tolerance` as a fraction of focal units (see method_rules.yaml
    `cross_check_agreement_tolerance`)."""
    if focal_units <= 0:
        return False
    return abs(rate_a - rate_b) > tolerance
