"""Structural discrete-choice engine: nested logit over a choice set that always
includes a "no-purchase / keep current" outside option.

This module owns the *deterministic* math (utility -> choice probabilities) used by
both the "native" maximum-likelihood estimator below and, when the `structural` extra
is installed, thin wrappers around `pyblp` (aggregate share+price data) and
`pylogit`/`xlogit` (individual-level data). All three paths converge on the same
`FittedChoiceModel` container so `counterfactual.py` never needs to know which engine
produced it.

No function in this module ever returns a bare float presented as "the answer" -- it
returns model objects that `counterfactual.py` turns into a `CannibalizationResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

NO_PURCHASE_ID = "__no_purchase__"
OUTSIDE_NEST = "__outside__"
_LAMBDA_FLOOR = 1e-3
Engine = Literal["native_mle", "pyblp", "pylogit", "xlogit"]


@dataclass(frozen=True)
class UtilityCoefficients:
    """The linear-in-parameters utility spec from the build plan:

    U = beta0_j + beta_loyalty * Loyalty(i, brand_j) + beta_price * Price(j,t)/Income(i)
        + beta_features * FeatureMatch(i,j) + beta_comp * CompetitorPressure(j,t)
        + beta_macro * MacroIndex(region,t) + eps
    """

    intercepts: dict[str, float]
    beta_loyalty: float
    beta_price: float
    beta_features: float
    beta_comp: float
    beta_macro: float


@dataclass(frozen=True)
class NestedLogitDesign:
    """Domain-agnostic design matrices for a nested-logit choice problem.

    Row i is one observation (an individual choice occasion, or a segment x period
    aggregate row). Column j is one alternative, product_ids[-1] is always the
    no-purchase outside option.
    """

    product_ids: tuple[str, ...]
    nest_of_product: dict[str, str]
    segment_ids: tuple[str, ...]
    market_size: npt.NDArray[np.float64]  # (N,)
    loyalty: npt.NDArray[np.float64]  # (N, J)
    price_over_income: npt.NDArray[np.float64]  # (N, J)
    feature_match: npt.NDArray[np.float64]  # (N, J)
    competitor_pressure: npt.NDArray[np.float64]  # (N, J)
    macro_index: npt.NDArray[np.float64]  # (N, J)
    choice_index: npt.NDArray[np.float64] | None = (
        None  # (N,) observed chosen column -- individual-level data
    )
    choice_weights: npt.NDArray[np.float64] | None = (
        None  # (N, J) observed counts/units -- aggregate/grouped data
    )

    def __post_init__(self) -> None:
        n, j = len(self.segment_ids), len(self.product_ids)
        for name, arr in (
            ("loyalty", self.loyalty),
            ("price_over_income", self.price_over_income),
            ("feature_match", self.feature_match),
            ("competitor_pressure", self.competitor_pressure),
            ("macro_index", self.macro_index),
        ):
            if arr.shape != (n, j):
                raise ValueError(f"{name} must have shape ({n}, {j}), got {arr.shape}")
        if self.market_size.shape != (n,):
            raise ValueError(f"market_size must have shape ({n},), got {self.market_size.shape}")
        if self.product_ids[-1] != NO_PURCHASE_ID:
            raise ValueError("the last product_id must be NO_PURCHASE_ID (outside option)")
        if self.nest_of_product.get(NO_PURCHASE_ID) != OUTSIDE_NEST:
            raise ValueError(f"nest_of_product[{NO_PURCHASE_ID!r}] must be OUTSIDE_NEST")
        if self.choice_weights is not None and self.choice_weights.shape != (n, j):
            raise ValueError(
                f"choice_weights must have shape ({n}, {j}), got {self.choice_weights.shape}"
            )

    @property
    def n_obs(self) -> int:
        return len(self.segment_ids)

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    def product_index(self, product_id: str) -> int:
        return self.product_ids.index(product_id)

    def nests(self) -> tuple[str, ...]:
        seen: list[str] = []
        for pid in self.product_ids:
            n = self.nest_of_product[pid]
            if n not in seen:
                seen.append(n)
        return tuple(seen)


def deterministic_utility(
    design: NestedLogitDesign,
    coefficients: UtilityCoefficients,
    masked_products: frozenset[str] = frozenset(),
) -> npt.NDArray[np.float64]:
    """Compute V[i, j] for every observation/alternative. Masked products get -inf,
    which is how "focal removed" counterfactuals are constructed downstream.
    """
    intercepts = np.array([coefficients.intercepts.get(pid, 0.0) for pid in design.product_ids])
    v = (
        intercepts[None, :]
        + coefficients.beta_loyalty * design.loyalty
        + coefficients.beta_price * design.price_over_income
        + coefficients.beta_features * design.feature_match
        + coefficients.beta_comp * design.competitor_pressure
        + coefficients.beta_macro * design.macro_index
    )
    if masked_products:
        mask_cols = [i for i, pid in enumerate(design.product_ids) if pid in masked_products]
        v = v.copy()
        v[:, mask_cols] = -np.inf
    return np.asarray(v, dtype=np.float64)


def nested_logit_probabilities(
    v: npt.NDArray[np.float64], design: NestedLogitDesign, lambdas: dict[str, float]
) -> npt.NDArray[np.float64]:
    """The standard McFadden nested-logit choice probability formula.

    P(i,n) = exp(lambda_n * IV_i,n) / sum_n' exp(lambda_n' * IV_i,n')
    P(i,j|n) = exp(V_ij/lambda_n) / sum_{j' in n} exp(V_ij'/lambda_n)
    P(i,j) = P(i,n(j)) * P(i,j|n(j))

    Numerically stable via logsumexp; a fully-masked (all -inf) nest correctly
    contributes zero probability to both levels instead of raising/NaN-ing.
    """
    nests = design.nests()
    n_obs = design.n_obs
    nest_log_prob = np.full((n_obs, len(nests)), -np.inf)
    within_nest_log_prob = np.full_like(v, -np.inf)

    for n_idx, nest in enumerate(nests):
        lam = lambdas[nest]
        cols = [
            i for i, pid in enumerate(design.product_ids) if design.nest_of_product[pid] == nest
        ]
        v_nest = v[:, cols]
        scaled = v_nest / lam
        # rows that are entirely -inf/lam stay -inf; logsumexp handles that gracefully
        log_denom = logsumexp(scaled, axis=1)
        within_nest_log_prob[:, cols] = scaled - log_denom[:, None]
        # inclusive value IV_i,n = log_denom (it's already log sum exp(V/lambda))
        nest_log_prob[:, n_idx] = lam * log_denom

    nest_log_denom = logsumexp(nest_log_prob, axis=1)
    nest_log_prob = nest_log_prob - nest_log_denom[:, None]

    log_p = np.full_like(v, -np.inf)
    for n_idx, nest in enumerate(nests):
        cols = [
            i for i, pid in enumerate(design.product_ids) if design.nest_of_product[pid] == nest
        ]
        log_p[:, cols] = nest_log_prob[:, [n_idx]] + within_nest_log_prob[:, cols]

    p = np.exp(log_p)
    # guard against tiny floating-point drift so rows always sum to exactly 1
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return np.asarray(p / row_sums, dtype=np.float64)


@dataclass
class FittedChoiceModel:
    """A fitted structural choice model, engine-agnostic from here on."""

    design: NestedLogitDesign
    coefficients: UtilityCoefficients
    lambdas: dict[str, float]
    engine: Engine
    log_likelihood: float | None = None
    coefficient_cov: npt.NDArray[np.float64] | None = None  # for bootstrap / delta-method CIs
    param_layout: _ParamLayout | None = field(default=None, repr=False)

    def choice_probabilities(
        self, masked_products: frozenset[str] = frozenset()
    ) -> npt.NDArray[np.float64]:
        v = deterministic_utility(self.design, self.coefficients, masked_products)
        return nested_logit_probabilities(v, self.design, self.lambdas)

    def resample_coefficients(self, rng: np.random.Generator) -> FittedChoiceModel:
        """Draw one coefficient vector from the asymptotic sampling distribution and
        return a new FittedChoiceModel with those coefficients (design unchanged).
        Used by `uncertainty.bootstrap_ci`.
        """
        if self.coefficient_cov is None or self.param_layout is None:
            raise ValueError("coefficient_cov/param_layout required to resample; refit with cov.")
        theta_hat = self.param_layout.pack(self.coefficients, self.lambdas)
        theta_draw = rng.multivariate_normal(theta_hat, self.coefficient_cov)
        coefficients, lambdas = self.param_layout.unpack(theta_draw)
        return FittedChoiceModel(
            design=self.design,
            coefficients=coefficients,
            lambdas=lambdas,
            engine=self.engine,
            log_likelihood=None,
            coefficient_cov=self.coefficient_cov,
            param_layout=self.param_layout,
        )


@dataclass(frozen=True)
class _ParamLayout:
    """Maps a flat optimizer vector theta <-> (UtilityCoefficients, lambdas)."""

    free_product_ids: tuple[str, ...]  # all products except NO_PURCHASE_ID (reference)
    free_nests: tuple[str, ...]  # all nests except OUTSIDE_NEST (lambda fixed at 1)

    @property
    def n_params(self) -> int:
        return len(self.free_product_ids) + 5 + len(self.free_nests)

    def pack(
        self, coefficients: UtilityCoefficients, lambdas: dict[str, float]
    ) -> npt.NDArray[np.float64]:
        intercepts = [coefficients.intercepts.get(pid, 0.0) for pid in self.free_product_ids]
        betas = [
            coefficients.beta_loyalty,
            coefficients.beta_price,
            coefficients.beta_features,
            coefficients.beta_comp,
            coefficients.beta_macro,
        ]
        raw_lambdas = [logit(np.clip(lambdas[n], _LAMBDA_FLOOR, 1 - 1e-9)) for n in self.free_nests]
        return np.array(intercepts + betas + raw_lambdas, dtype=float)

    def unpack(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[UtilityCoefficients, dict[str, float]]:
        n_int = len(self.free_product_ids)
        intercepts = dict(zip(self.free_product_ids, theta[:n_int], strict=True))
        intercepts[NO_PURCHASE_ID] = 0.0
        beta_loyalty, beta_price, beta_features, beta_comp, beta_macro = theta[n_int : n_int + 5]
        raw_lambdas = theta[n_int + 5 :]
        lambdas = {
            n: float(_LAMBDA_FLOOR + (1 - _LAMBDA_FLOOR) * expit(r))
            for n, r in zip(self.free_nests, raw_lambdas, strict=True)
        }
        lambdas[OUTSIDE_NEST] = 1.0
        coefficients = UtilityCoefficients(
            intercepts=intercepts,
            beta_loyalty=float(beta_loyalty),
            beta_price=float(beta_price),
            beta_features=float(beta_features),
            beta_comp=float(beta_comp),
            beta_macro=float(beta_macro),
        )
        return coefficients, lambdas


def _negative_log_likelihood(
    theta: npt.NDArray[np.float64], design: NestedLogitDesign, layout: _ParamLayout
) -> float:
    """Individual-level data (`choice_index`): standard per-row chosen-alternative MLE.
    Aggregate/grouped data (`choice_weights`): the algebraically equivalent grouped
    multinomial-logit MLE, LL = sum_i sum_j weight_ij * log(P_ij) -- weight_ij is the
    observed unit count for market i, product j (e.g. from scanner/shipment data),
    reducing to the individual-level case when weights are one-hot per row.
    """
    coefficients, lambdas = layout.unpack(theta)
    v = deterministic_utility(design, coefficients)
    p = nested_logit_probabilities(v, design, lambdas)
    log_p = np.log(np.clip(p, 1e-12, 1.0))
    if design.choice_index is not None:
        chosen_log_p = log_p[np.arange(design.n_obs), design.choice_index]
        return float(-np.sum(chosen_log_p))
    assert design.choice_weights is not None
    return float(-np.sum(design.choice_weights * log_p))


def fit_nested_logit(
    design: NestedLogitDesign,
    initial: UtilityCoefficients | None = None,
    initial_lambdas: dict[str, float] | None = None,
) -> FittedChoiceModel:
    """Native maximum-likelihood nested-logit estimator (no external structural lib
    required). Used when `pyblp`/`pylogit`/`xlogit` are unavailable, and as the
    default/reference implementation validated by the known-answer fixture tests.
    """
    if design.choice_index is None and design.choice_weights is None:
        raise ValueError(
            "fit_nested_logit requires design.choice_index (individual-level observed "
            "choices) or design.choice_weights (aggregate/grouped unit counts)"
        )

    free_products = tuple(pid for pid in design.product_ids if pid != NO_PURCHASE_ID)
    free_nests = tuple(n for n in design.nests() if n != OUTSIDE_NEST)
    layout = _ParamLayout(free_product_ids=free_products, free_nests=free_nests)

    if initial is None:
        initial = UtilityCoefficients(
            intercepts=dict.fromkeys(free_products, 0.0),
            beta_loyalty=0.1,
            beta_price=-0.1,
            beta_features=0.1,
            beta_comp=-0.1,
            beta_macro=0.1,
        )
    if initial_lambdas is None:
        initial_lambdas = dict.fromkeys(free_nests, 0.7)
        initial_lambdas[OUTSIDE_NEST] = 1.0

    theta0 = layout.pack(initial, initial_lambdas)
    result = minimize(
        _negative_log_likelihood,
        theta0,
        args=(design, layout),
        method="BFGS",
        options={"maxiter": 300, "gtol": 1e-5},
    )
    coefficients, lambdas = layout.unpack(result.x)
    cov = result.hess_inv if isinstance(result.hess_inv, np.ndarray) else None
    return FittedChoiceModel(
        design=design,
        coefficients=coefficients,
        lambdas=lambdas,
        engine="native_mle",
        log_likelihood=-float(result.fun),
        coefficient_cov=cov,
        param_layout=layout,
    )


def fit_via_pylogit(design: NestedLogitDesign, **pylogit_kwargs: object) -> FittedChoiceModel:
    """Thin wrapper around `pylogit` for individual-level nested logit. Requires the
    `structural` extra. Falls back is the caller's responsibility (see
    `src/agents/causal_agent.py`), not this function's -- it either fits or raises.
    """
    try:
        import pylogit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when extra installed
        raise ImportError(
            "pylogit is not installed; install the 'structural' extra "
            "(`uv pip install '.[structural]'`) or use fit_nested_logit() instead."
        ) from exc
    # NOTE: pylogit's long-format API requires building a pandas long DataFrame and an
    # explicit specification/name dict per nest. That wiring is intentionally left for
    # the integration point where real individual-level data is available (Phase 2+),
    # since its call shape depends on the adapter's column names. The native estimator
    # above implements the identical statistical model and is the default engine.
    raise NotImplementedError(
        "pylogit wiring is adapter-dependent; use fit_nested_logit() (same model, "
        "native implementation) until a concrete pylogit long-format mapping is added."
    )


def fit_via_xlogit(design: NestedLogitDesign, **xlogit_kwargs: object) -> FittedChoiceModel:
    """Thin wrapper around `xlogit` for individual-level mixed logit. See
    `fit_via_pylogit` docstring -- same status.
    """
    try:
        import xlogit  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xlogit is not installed; install the 'structural' extra "
            "(`uv pip install '.[structural]'`) or use fit_nested_logit() instead."
        ) from exc
    raise NotImplementedError(
        "xlogit wiring is adapter-dependent; use fit_nested_logit() (same model "
        "family, native implementation) until a concrete xlogit mapping is added."
    )


def fit_via_pyblp(design: NestedLogitDesign, **pyblp_kwargs: object) -> FittedChoiceModel:
    """Thin wrapper around `pyblp` for aggregate share+price BLP/nested logit. See
    `fit_via_pylogit` docstring -- same status.
    """
    try:
        import pyblp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyblp is not installed; install the 'structural' extra "
            "(`uv pip install '.[structural]'`) or use fit_nested_logit() instead."
        ) from exc
    raise NotImplementedError(
        "pyblp wiring is adapter-dependent; use fit_nested_logit() (same model "
        "family, native implementation) until a concrete pyblp formulation is added."
    )
