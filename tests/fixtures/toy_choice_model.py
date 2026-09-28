"""A hand-constructed (not fitted) nested-logit fixture with a known analytic
counterfactual answer, used to gate the causal engine (build plan section 6.5).

Products: A1 (focal), A2 (within-brand incumbent, same nest as A1), B1 (competitor,
different nest). All covariates are zero so utility reduces to the alternative-specific
intercept, which keeps the by-hand arithmetic in `tests/unit/test_counterfactual.py`
tractable while still exercising the full nested-logit probability formula and the
two-level (within-nest, then across-nest) redistribution it implies.
"""

from __future__ import annotations

import numpy as np

from src.causal.nested_logit import (
    NO_PURCHASE_ID,
    OUTSIDE_NEST,
    FittedChoiceModel,
    NestedLogitDesign,
    UtilityCoefficients,
)

FOCAL_ID = "A1"
WITHIN_BRAND_INCUMBENT_ID = "A2"
COMPETITOR_INCUMBENT_ID = "B1"
BRAND_A_NEST = "BrandA"
BRAND_B_NEST = "BrandB"

INTERCEPTS = {FOCAL_ID: 1.0, WITHIN_BRAND_INCUMBENT_ID: 0.5, COMPETITOR_INCUMBENT_ID: 0.3}
LAMBDA_BRAND_A = 0.5
LAMBDA_BRAND_B = 1.0
MARKET_SIZE = 1000.0


def build_single_segment_design(market_size: float = MARKET_SIZE) -> NestedLogitDesign:
    product_ids = (FOCAL_ID, WITHIN_BRAND_INCUMBENT_ID, COMPETITOR_INCUMBENT_ID, NO_PURCHASE_ID)
    nest_of_product = {
        FOCAL_ID: BRAND_A_NEST,
        WITHIN_BRAND_INCUMBENT_ID: BRAND_A_NEST,
        COMPETITOR_INCUMBENT_ID: BRAND_B_NEST,
        NO_PURCHASE_ID: OUTSIDE_NEST,
    }
    n, j = 1, len(product_ids)
    zeros = np.zeros((n, j))
    return NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=("seg1",),
        market_size=np.array([market_size]),
        loyalty=zeros,
        price_over_income=zeros,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
    )


def build_single_segment_fitted_model(market_size: float = MARKET_SIZE) -> FittedChoiceModel:
    design = build_single_segment_design(market_size)
    coefficients = UtilityCoefficients(
        intercepts=dict(INTERCEPTS),
        beta_loyalty=0.0,
        beta_price=0.0,
        beta_features=0.0,
        beta_comp=0.0,
        beta_macro=0.0,
    )
    lambdas = {BRAND_A_NEST: LAMBDA_BRAND_A, BRAND_B_NEST: LAMBDA_BRAND_B, OUTSIDE_NEST: 1.0}
    return FittedChoiceModel(
        design=design, coefficients=coefficients, lambdas=lambdas, engine="native_mle"
    )


def build_two_segment_design(
    market_sizes: tuple[float, float] = (600.0, 400.0),
) -> NestedLogitDesign:
    """Same product/nest structure as the single-segment fixture, but two segments with
    different loyalty levels, to exercise per-segment aggregation and dominant-source
    classification differing by segment.
    """
    product_ids = (FOCAL_ID, WITHIN_BRAND_INCUMBENT_ID, COMPETITOR_INCUMBENT_ID, NO_PURCHASE_ID)
    nest_of_product = {
        FOCAL_ID: BRAND_A_NEST,
        WITHIN_BRAND_INCUMBENT_ID: BRAND_A_NEST,
        COMPETITOR_INCUMBENT_ID: BRAND_B_NEST,
        NO_PURCHASE_ID: OUTSIDE_NEST,
    }
    n, j = 2, len(product_ids)
    loyalty = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],  # segment 1: loyal to brand A
            [0.0, 0.0, 1.0, 0.0],  # segment 2: loyal to brand B
        ]
    )
    zeros = np.zeros((n, j))
    return NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=("loyal_a", "loyal_b"),
        market_size=np.array(market_sizes),
        loyalty=loyalty,
        price_over_income=zeros,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
    )


def build_two_segment_fitted_model(
    market_sizes: tuple[float, float] = (600.0, 400.0),
) -> FittedChoiceModel:
    design = build_two_segment_design(market_sizes)
    coefficients = UtilityCoefficients(
        intercepts=dict(INTERCEPTS),
        beta_loyalty=1.5,
        beta_price=0.0,
        beta_features=0.0,
        beta_comp=0.0,
        beta_macro=0.0,
    )
    lambdas = {BRAND_A_NEST: LAMBDA_BRAND_A, BRAND_B_NEST: LAMBDA_BRAND_B, OUTSIDE_NEST: 1.0}
    return FittedChoiceModel(
        design=design, coefficients=coefficients, lambdas=lambdas, engine="native_mle"
    )
