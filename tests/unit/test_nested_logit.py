import math

import numpy as np
import pytest

from src.causal.nested_logit import (
    NO_PURCHASE_ID,
    OUTSIDE_NEST,
    UtilityCoefficients,
    deterministic_utility,
    nested_logit_probabilities,
)
from tests.fixtures.toy_choice_model import (
    BRAND_A_NEST,
    BRAND_B_NEST,
    COMPETITOR_INCUMBENT_ID,
    FOCAL_ID,
    INTERCEPTS,
    LAMBDA_BRAND_A,
    LAMBDA_BRAND_B,
    WITHIN_BRAND_INCUMBENT_ID,
    build_single_segment_design,
)


def _independent_nested_logit_probs() -> dict[str, float]:
    """Reference implementation computed directly from the McFadden formula with
    plain math.exp/log, independent of nested_logit_probabilities, to catch bugs in
    the vectorized implementation rather than confirming its own arithmetic.
    """
    v_a1, v_a2, v_b1, v_out = (
        INTERCEPTS[FOCAL_ID],
        INTERCEPTS[WITHIN_BRAND_INCUMBENT_ID],
        INTERCEPTS[COMPETITOR_INCUMBENT_ID],
        0.0,
    )

    iv_a = math.log(math.exp(v_a1 / LAMBDA_BRAND_A) + math.exp(v_a2 / LAMBDA_BRAND_A))
    iv_b = math.log(math.exp(v_b1 / LAMBDA_BRAND_B))
    iv_out = math.log(math.exp(v_out / 1.0))

    nu_a = LAMBDA_BRAND_A * iv_a
    nu_b = LAMBDA_BRAND_B * iv_b
    nu_out = 1.0 * iv_out

    denom = math.exp(nu_a) + math.exp(nu_b) + math.exp(nu_out)
    p_nest_a = math.exp(nu_a) / denom
    p_nest_b = math.exp(nu_b) / denom
    p_nest_out = math.exp(nu_out) / denom

    p_a1_given_a = math.exp(v_a1 / LAMBDA_BRAND_A) / (
        math.exp(v_a1 / LAMBDA_BRAND_A) + math.exp(v_a2 / LAMBDA_BRAND_A)
    )
    p_a2_given_a = 1 - p_a1_given_a

    return {
        FOCAL_ID: p_nest_a * p_a1_given_a,
        WITHIN_BRAND_INCUMBENT_ID: p_nest_a * p_a2_given_a,
        COMPETITOR_INCUMBENT_ID: p_nest_b * 1.0,
        NO_PURCHASE_ID: p_nest_out * 1.0,
    }


def test_nested_logit_probabilities_match_independent_reference() -> None:
    design = build_single_segment_design()
    coefficients = UtilityCoefficients(
        intercepts=dict(INTERCEPTS),
        beta_loyalty=0,
        beta_price=0,
        beta_features=0,
        beta_comp=0,
        beta_macro=0,
    )
    lambdas = {BRAND_A_NEST: LAMBDA_BRAND_A, BRAND_B_NEST: LAMBDA_BRAND_B, OUTSIDE_NEST: 1.0}
    v = deterministic_utility(design, coefficients)
    p = nested_logit_probabilities(v, design, lambdas)

    expected = _independent_nested_logit_probs()
    for j, pid in enumerate(design.product_ids):
        assert p[0, j] == pytest.approx(expected[pid], abs=1e-10)


def test_probabilities_sum_to_one() -> None:
    design = build_single_segment_design()
    coefficients = UtilityCoefficients(
        intercepts=dict(INTERCEPTS),
        beta_loyalty=0,
        beta_price=0,
        beta_features=0,
        beta_comp=0,
        beta_macro=0,
    )
    lambdas = {BRAND_A_NEST: LAMBDA_BRAND_A, BRAND_B_NEST: LAMBDA_BRAND_B, OUTSIDE_NEST: 1.0}
    v = deterministic_utility(design, coefficients)
    p = nested_logit_probabilities(v, design, lambdas)
    assert p.sum(axis=1) == pytest.approx(1.0)


def test_masked_product_has_zero_probability() -> None:
    design = build_single_segment_design()
    coefficients = UtilityCoefficients(
        intercepts=dict(INTERCEPTS),
        beta_loyalty=0,
        beta_price=0,
        beta_features=0,
        beta_comp=0,
        beta_macro=0,
    )
    lambdas = {BRAND_A_NEST: LAMBDA_BRAND_A, BRAND_B_NEST: LAMBDA_BRAND_B, OUTSIDE_NEST: 1.0}
    v = deterministic_utility(design, coefficients, masked_products=frozenset({FOCAL_ID}))
    p = nested_logit_probabilities(v, design, lambdas)
    focal_idx = design.product_index(FOCAL_ID)
    assert p[0, focal_idx] == pytest.approx(0.0)
    assert p.sum(axis=1) == pytest.approx(1.0)


def test_design_rejects_wrong_shape() -> None:
    from src.causal.nested_logit import NestedLogitDesign

    with pytest.raises(ValueError):
        NestedLogitDesign(
            product_ids=(FOCAL_ID, NO_PURCHASE_ID),
            nest_of_product={FOCAL_ID: BRAND_A_NEST, NO_PURCHASE_ID: OUTSIDE_NEST},
            segment_ids=("seg1",),
            market_size=np.array([1.0]),
            loyalty=np.zeros((1, 1)),  # wrong shape: should be (1, 2)
            price_over_income=np.zeros((1, 2)),
            feature_match=np.zeros((1, 2)),
            competitor_pressure=np.zeros((1, 2)),
            macro_index=np.zeros((1, 2)),
        )
