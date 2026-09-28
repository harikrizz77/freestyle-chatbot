"""Phase 1 acceptance gate: the counterfactual engine must reproduce a hand-computed
known answer on a toy fixture, and the mass-balance identity must hold exactly.
"""

import math

import pytest

from src.causal.counterfactual import compute_counterfactual
from src.causal.uncertainty import MassBalanceError, assert_mass_balance
from tests.fixtures.toy_choice_model import (
    COMPETITOR_INCUMBENT_ID,
    FOCAL_ID,
    INTERCEPTS,
    LAMBDA_BRAND_A,
    LAMBDA_BRAND_B,
    MARKET_SIZE,
    WITHIN_BRAND_INCUMBENT_ID,
    build_single_segment_fitted_model,
    build_two_segment_fitted_model,
)


def _analytic_expected_units() -> dict[str, float]:
    """Independent, by-hand derivation of the known answer (see build plan 6.5)."""
    v_a1, v_a2, v_b1 = (
        INTERCEPTS[FOCAL_ID],
        INTERCEPTS[WITHIN_BRAND_INCUMBENT_ID],
        INTERCEPTS[COMPETITOR_INCUMBENT_ID],
    )

    def nest_utils(mask_a1: bool) -> tuple[float, float, float]:
        a1_term = 0.0 if mask_a1 else math.exp(v_a1 / LAMBDA_BRAND_A)
        iv_a = math.log(a1_term + math.exp(v_a2 / LAMBDA_BRAND_A))
        nu_a = LAMBDA_BRAND_A * iv_a
        nu_b = LAMBDA_BRAND_B * v_b1
        nu_out = 0.0
        return nu_a, nu_b, nu_out

    def probs(mask_a1: bool) -> dict[str, float]:
        nu_a, nu_b, nu_out = nest_utils(mask_a1)
        denom = math.exp(nu_a) + math.exp(nu_b) + math.exp(nu_out)
        p_nest_a = math.exp(nu_a) / denom
        p_nest_b = math.exp(nu_b) / denom
        p_nest_out = math.exp(nu_out) / denom
        if mask_a1:
            p_a1, p_a2 = 0.0, p_nest_a
        else:
            a1_term = math.exp(v_a1 / LAMBDA_BRAND_A)
            a2_term = math.exp(v_a2 / LAMBDA_BRAND_A)
            p_a1 = p_nest_a * a1_term / (a1_term + a2_term)
            p_a2 = p_nest_a * a2_term / (a1_term + a2_term)
        return {
            FOCAL_ID: p_a1,
            WITHIN_BRAND_INCUMBENT_ID: p_a2,
            COMPETITOR_INCUMBENT_ID: p_nest_b,
            "__no_purchase__": p_nest_out,
        }

    p_factual = probs(mask_a1=False)
    p_counterfactual = probs(mask_a1=True)
    return {pid: (p_counterfactual[pid] - p_factual[pid]) * MARKET_SIZE for pid in p_factual} | {
        "__focal_factual_units__": p_factual[FOCAL_ID] * MARKET_SIZE
    }


def test_known_answer_counterfactual_single_segment() -> None:
    fitted = build_single_segment_fitted_model()
    outcome = compute_counterfactual(fitted, FOCAL_ID)
    expected = _analytic_expected_units()

    assert outcome.total_focal_units == pytest.approx(expected["__focal_factual_units__"], rel=1e-9)
    assert outcome.incumbent_units[WITHIN_BRAND_INCUMBENT_ID] == pytest.approx(
        expected[WITHIN_BRAND_INCUMBENT_ID], rel=1e-9
    )
    assert outcome.incumbent_units[COMPETITOR_INCUMBENT_ID] == pytest.approx(
        expected[COMPETITOR_INCUMBENT_ID], rel=1e-9
    )
    assert outcome.incumbent_units["__no_purchase__"] == pytest.approx(
        expected["__no_purchase__"], rel=1e-9
    )

    # within-brand incumbent should absorb more of focal's demand than the
    # cross-nest competitor, because lambda_BrandA < 1 correlates same-nest options
    assert (
        outcome.incumbent_units[WITHIN_BRAND_INCUMBENT_ID]
        > outcome.incumbent_units[COMPETITOR_INCUMBENT_ID]
    )

    rate_within = outcome.cannibalization_rate(WITHIN_BRAND_INCUMBENT_ID)
    rate_competitor = outcome.cannibalization_rate(COMPETITOR_INCUMBENT_ID)
    overall = outcome.overall_cannibalization_rate(
        [WITHIN_BRAND_INCUMBENT_ID, COMPETITOR_INCUMBENT_ID]
    )
    assert 0.0 < rate_within < 1.0
    assert 0.0 < rate_competitor < 1.0
    assert overall == pytest.approx(rate_within + rate_competitor, rel=1e-9)

    seg = outcome.segments[0]
    assert seg.dominant_source == "within_brand"


def test_mass_balance_holds_exactly() -> None:
    fitted = build_single_segment_fitted_model()
    outcome = compute_counterfactual(fitted, FOCAL_ID)
    assert outcome.mass_balance.passed
    assert_mass_balance(outcome.mass_balance)  # must not raise


def test_mass_balance_detects_injected_bug() -> None:
    from src.schema.results import MassBalanceCheck

    broken = MassBalanceCheck(focal_units_removed=100.0, redistributed_units_total=80.0)
    assert not broken.passed
    with pytest.raises(MassBalanceError):
        assert_mass_balance(broken)


def test_two_segment_aggregation_and_dominant_source() -> None:
    fitted = build_two_segment_fitted_model()
    outcome = compute_counterfactual(fitted, FOCAL_ID)
    assert outcome.mass_balance.passed
    assert len(outcome.segments) == 2

    seg_results = outcome.segment_results([WITHIN_BRAND_INCUMBENT_ID, COMPETITOR_INCUMBENT_ID])
    assert {s.segment_id for s in seg_results} == {"loyal_a", "loyal_b"}
    loyal_a = next(s for s in seg_results if s.segment_id == "loyal_a")
    # segment loyal to brand A should show within-brand as the dominant redistribution source
    assert loyal_a.dominant_source == "within_brand"


def test_focal_not_in_design_raises() -> None:
    fitted = build_single_segment_fitted_model()
    with pytest.raises(ValueError):
        compute_counterfactual(fitted, "does_not_exist")
