from src.agents.causal_agent import CausalAgentResult
from src.agents.validator import validate
from src.schema.results import CannibalizationResult, MassBalanceCheck, ReducedFormResult
from src.schema.signals import ConcurrentShockAssessment
from src.schema.tools import MethodSelection


def _prelim(**overrides) -> CannibalizationResult:
    base = dict(
        focal_product_id="F1",
        overall_rate=0.3,
        ci_low=0.25,
        ci_high=0.35,
        method_used="blp_nested_logit",
        confidence="high",
        seed=42,
    )
    base.update(overrides)
    return CannibalizationResult(**base)


def _selection(**overrides) -> MethodSelection:
    base = dict(
        rule_id="blp_aggregate",
        primary_method="blp_nested_logit",
        cross_check_methods=[],
        segmentation="none",
    )
    base.update(overrides)
    return MethodSelection(**base)


def _no_shock() -> ConcurrentShockAssessment:
    return ConcurrentShockAssessment(level="none")


def test_mass_balance_failure_halts() -> None:
    prelim = _prelim(
        mass_balance=MassBalanceCheck(focal_units_removed=100.0, redistributed_units_total=50.0)
    )
    causal_result = CausalAgentResult(prelim, None, [], [])
    final = validate(causal_result, _selection(), _no_shock())
    assert final.halted is True
    assert "mass balance" in final.halt_reason.lower()
    assert final.confidence == "low"


def test_mass_balance_pass_does_not_halt() -> None:
    prelim = _prelim(
        mass_balance=MassBalanceCheck(focal_units_removed=100.0, redistributed_units_total=100.0)
    )
    causal_result = CausalAgentResult(prelim, None, [], [])
    final = validate(causal_result, _selection(), _no_shock())
    assert final.halted is False
    assert final.confidence == "high"


def test_confirmed_shock_without_covariate_halts() -> None:
    causal_result = CausalAgentResult(_prelim(), None, [], [])
    shock = ConcurrentShockAssessment(level="confirmed", covariate_included=False, rationale="x")
    final = validate(causal_result, _selection(), shock)
    assert final.halted is True
    assert "confirmed_competitor_or_macro_shock_not_modeled" in final.confounders_flagged


def test_confirmed_shock_with_covariate_downgrades_instead_of_halting() -> None:
    """The whole point of wiring competitor/macro data into the math: a confirmed
    shock that *was* folded into the fit is downgraded to low confidence, not halted.
    """
    causal_result = CausalAgentResult(_prelim(), None, [], [], used_market_context=True)
    shock = ConcurrentShockAssessment(level="confirmed", covariate_included=True, rationale="x")
    final = validate(causal_result, _selection(), shock)
    assert final.halted is False
    assert final.confidence == "low"
    assert "confirmed_competitor_or_macro_shock" in final.confounders_flagged
    assert final.overall_rate == causal_result.preliminary_result.overall_rate  # number untouched


def test_suspected_shock_caps_confidence_to_medium() -> None:
    causal_result = CausalAgentResult(_prelim(), None, [], [])
    shock = ConcurrentShockAssessment(level="suspected", rationale="x")
    final = validate(causal_result, _selection(), shock)
    assert final.halted is False
    assert final.confidence == "medium"


def test_cross_check_disagreement_caps_confidence_to_medium() -> None:
    causal_result = CausalAgentResult(
        _prelim(overall_rate=0.3), None, [], [], cross_check_overall_rate=0.6
    )
    final = validate(causal_result, _selection(), _no_shock(), cross_check_tolerance=0.10)
    assert final.confidence == "medium"


def test_confidence_ceiling_from_method_selection_is_respected() -> None:
    causal_result = CausalAgentResult(_prelim(), None, [], [])
    final = validate(causal_result, _selection(confidence_ceiling="low"), _no_shock())
    assert final.confidence == "low"


def test_parallel_trends_failure_forces_low_confidence() -> None:
    failing_result = ReducedFormResult(
        method="did",
        incumbent_product_id="I1",
        actual_units=100.0,
        counterfactual_units=120.0,
        impact_units=-20.0,
        ci_low=-30.0,
        ci_high=-10.0,
        parallel_trends_passed=False,
    )
    causal_result = CausalAgentResult(_prelim(method_used="did"), None, [failing_result], [])
    final = validate(causal_result, _selection(primary_method="did"), _no_shock())
    assert final.confidence == "low"
