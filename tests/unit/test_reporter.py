from src.agents.reporter import narrate
from src.schema.results import (
    CannibalizationResult,
    MassBalanceCheck,
    SegmentResult,
    ValidatorFlag,
)
from src.schema.tools import DataProfile, MethodSelection


def _base_result(**overrides) -> CannibalizationResult:
    base = dict(
        focal_product_id="1002",
        overall_rate=0.324,
        ci_low=0.305,
        ci_high=0.343,
        segments=[
            SegmentResult(
                segment_id="store_1__2024-01-07",
                cannibalization_rate=0.324,
                ci_low=0.324,
                ci_high=0.324,
                dominant_source="net_new_demand",
            )
        ],
        method_used="blp_nested_logit",
        cross_check_methods=["synthetic_control", "causal_impact"],
        confidence="high",
        confounders_flagged=[],
        validator_flags=[ValidatorFlag(check="mass_balance", passed=True, severity="info")],
        mass_balance=MassBalanceCheck(focal_units_removed=150.0, redistributed_units_total=150.0),
        seed=42,
        halted=False,
    )
    base.update(overrides)
    return CannibalizationResult(**base)


def _data_profile() -> DataProfile:
    return DataProfile(
        data_level="aggregate",
        price_variation="sufficient",
        donor_pool="weak",
        pre_periods=1,
        segments_estimable=False,
        n_products=3,
        n_regions=1,
        coverage_notes="6 sales rows across 3 products and 1 region",
    )


def _method_selection() -> MethodSelection:
    return MethodSelection(
        rule_id="blp_aggregate",
        primary_method="blp_nested_logit",
        cross_check_methods=["synthetic_control", "causal_impact"],
        segmentation="region_or_price_tier",
        notes="Classic electronics/CPG-shipment case (e.g. the iPhone demo).",
    )


def test_narrate_only_changes_the_narrative_field() -> None:
    result = _base_result()
    narrated = narrate(result, data_profile=_data_profile(), method_selection=_method_selection())

    without_narrative = narrated.model_copy(update={"narrative": ""})
    original_without_narrative = result.model_copy(update={"narrative": ""})
    assert without_narrative == original_without_narrative
    assert narrated.narrative != ""


def test_template_narrative_is_elaborate_and_explains_methodology() -> None:
    result = _base_result()
    narrated = narrate(result, data_profile=_data_profile(), method_selection=_method_selection())
    text = narrated.narrative

    # methodology transparency: mentions why this method, and what it actually does
    assert "## Why we used this method" in text
    assert "BLP" in text or "store-level sales" in text
    assert "## What data we used" in text
    assert "## Bottom line" in text
    assert "## How much should you trust this?" in text
    # numbers are quoted, not re-derived
    assert "32.4%" in text
    assert "30.5%" in text and "34.3%" in text
    # segment breakdown explained in plain language, not just raw field names
    assert "store_1__2024-01-07" in text
    assert "genuinely new" in text or "new demand" in text
    # much longer than the old one-liner template
    assert len(text) > 800


def test_narrate_without_context_still_produces_a_full_narrative() -> None:
    result = _base_result()
    narrated = narrate(result)  # no data_profile/method_selection supplied
    assert "## Bottom line" in narrated.narrative
    assert "32.4%" in narrated.narrative


def test_halted_result_explains_why_in_plain_language() -> None:
    result = _base_result(
        confidence="low",
        halted=True,
        halt_reason=(
            "a confirmed concurrent competitor/macro shock overlaps the analysis window and was "
            "not modeled as a covariate"
        ),
        confounders_flagged=["confirmed_competitor_or_macro_shock_not_modeled"],
        validator_flags=[
            ValidatorFlag(
                check="concurrent_shock", passed=False, detail="RivalCo launch", severity="blocking"
            )
        ],
    )
    narrated = narrate(result, method_selection=_method_selection())

    assert "stopped" in narrated.narrative.lower()
    assert "concurrent_shock" in narrated.narrative
    assert narrated.overall_rate == result.overall_rate  # numbers still untouched
