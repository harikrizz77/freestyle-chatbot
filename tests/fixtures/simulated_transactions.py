"""Simulated individual-level (transaction) choice data generated *from* the nested
logit model itself, with known true parameters. Used to validate that
`fit_nested_logit`'s MLE actually recovers the generating parameters -- the closest
proxy available in this environment to fitting real Dominick's scanner data (build
plan section 6.5 asks for a holdout hit-rate / log-likelihood report on real data;
network access to download and license Nielsen/Dominick's panels is out of scope for
this sandbox, so this simulated-ground-truth fixture is the practical substitute: it
tests the identical statistical machinery end-to-end).
"""

from __future__ import annotations

import numpy as np

from src.causal.nested_logit import (
    NO_PURCHASE_ID,
    OUTSIDE_NEST,
    NestedLogitDesign,
    UtilityCoefficients,
    deterministic_utility,
    nested_logit_probabilities,
)

FOCAL_ID = "A1"
INCUMBENT_ID = "A2"
COMPETITOR_ID = "B1"
BRAND_A_NEST = "BrandA"
BRAND_B_NEST = "BrandB"

TRUE_COEFFICIENTS = UtilityCoefficients(
    intercepts={FOCAL_ID: 0.8, INCUMBENT_ID: 0.4, COMPETITOR_ID: 0.2},
    beta_loyalty=1.2,
    beta_price=-1.5,
    beta_features=0.0,
    beta_comp=0.0,
    beta_macro=0.0,
)
TRUE_LAMBDAS = {BRAND_A_NEST: 0.6, BRAND_B_NEST: 1.0, OUTSIDE_NEST: 1.0}


def simulate_transaction_design(n: int = 2500, seed: int = 0) -> NestedLogitDesign:
    rng = np.random.default_rng(seed)
    product_ids = (FOCAL_ID, INCUMBENT_ID, COMPETITOR_ID, NO_PURCHASE_ID)
    nest_of_product = {
        FOCAL_ID: BRAND_A_NEST,
        INCUMBENT_ID: BRAND_A_NEST,
        COMPETITOR_ID: BRAND_B_NEST,
        NO_PURCHASE_ID: OUTSIDE_NEST,
    }
    j = len(product_ids)

    # half the population is brand-A loyal (loyalty=1 on A1/A2, 0 on B1/no-purchase),
    # the rest brand-B loyal; independent per-alternative price/income draws
    brand_a_loyal = rng.integers(0, 2, size=n).astype(float)
    loyalty = np.zeros((n, j))
    loyalty[:, 0] = brand_a_loyal  # A1
    loyalty[:, 1] = brand_a_loyal  # A2
    loyalty[:, 2] = 1 - brand_a_loyal  # B1

    price_over_income = rng.uniform(0.0, 1.0, size=(n, j))
    price_over_income[:, 3] = 0.0  # no-purchase has no price

    zeros = np.zeros((n, j))
    design_for_utility = NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=tuple(f"obs_{i}" for i in range(n)),
        market_size=np.ones(n),
        loyalty=loyalty,
        price_over_income=price_over_income,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
    )

    v = deterministic_utility(design_for_utility, TRUE_COEFFICIENTS)
    p = nested_logit_probabilities(v, design_for_utility, TRUE_LAMBDAS)

    choice_index = np.array([rng.choice(j, p=p[i]) for i in range(n)], dtype=int)

    return NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=design_for_utility.segment_ids,
        market_size=design_for_utility.market_size,
        loyalty=loyalty,
        price_over_income=price_over_income,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
        choice_index=choice_index,
    )


def simulate_aggregate_design(
    n_markets: int = 400, units_per_market: int = 200, seed: int = 0
) -> NestedLogitDesign:
    """Grouped/aggregate analogue of `simulate_transaction_design`: each row is a
    "market" (e.g. a region x period cell) with observed unit *counts* per product
    (`choice_weights`) instead of one individual choice per row -- the data shape
    `blp_nested_logit`-style aggregate share+price sources actually provide.
    """
    rng = np.random.default_rng(seed)
    product_ids = (FOCAL_ID, INCUMBENT_ID, COMPETITOR_ID, NO_PURCHASE_ID)
    nest_of_product = {
        FOCAL_ID: BRAND_A_NEST,
        INCUMBENT_ID: BRAND_A_NEST,
        COMPETITOR_ID: BRAND_B_NEST,
        NO_PURCHASE_ID: OUTSIDE_NEST,
    }
    j = len(product_ids)

    brand_a_loyal_share = rng.uniform(0.2, 0.8, size=n_markets)
    loyalty = np.zeros((n_markets, j))
    loyalty[:, 0] = brand_a_loyal_share
    loyalty[:, 1] = brand_a_loyal_share
    loyalty[:, 2] = 1 - brand_a_loyal_share

    price_over_income = rng.uniform(0.0, 1.0, size=(n_markets, j))
    price_over_income[:, 3] = 0.0

    zeros = np.zeros((n_markets, j))
    design_for_utility = NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=tuple(f"market_{i}" for i in range(n_markets)),
        market_size=np.full(n_markets, float(units_per_market)),
        loyalty=loyalty,
        price_over_income=price_over_income,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
    )
    v = deterministic_utility(design_for_utility, TRUE_COEFFICIENTS)
    p = nested_logit_probabilities(v, design_for_utility, TRUE_LAMBDAS)
    counts = np.array(
        [rng.multinomial(units_per_market, p[i]) for i in range(n_markets)], dtype=float
    )

    return NestedLogitDesign(
        product_ids=product_ids,
        nest_of_product=nest_of_product,
        segment_ids=design_for_utility.segment_ids,
        market_size=design_for_utility.market_size,
        loyalty=loyalty,
        price_over_income=price_over_income,
        feature_match=zeros,
        competitor_pressure=zeros,
        macro_index=zeros,
        choice_weights=counts,
    )
