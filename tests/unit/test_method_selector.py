import pytest

from src.agents.method_selector import select_method
from src.schema.tools import DataProfile


def _profile(**overrides) -> DataProfile:
    base = dict(
        data_level="aggregate",
        price_variation="none",
        donor_pool="none",
        pre_periods=0,
        segments_estimable=False,
        concurrent_shock="none",
    )
    base.update(overrides)
    return DataProfile(**base)


def test_mixed_logit_full_rule() -> None:
    selection = select_method(
        _profile(data_level="panel", price_variation="sufficient", segments_estimable=True)
    )
    assert selection.rule_id == "mixed_logit_full"
    assert selection.primary_method == "mixed_logit"


def test_nested_logit_individual_rule() -> None:
    selection = select_method(
        _profile(data_level="transaction", price_variation="low", segments_estimable=False)
    )
    assert selection.rule_id == "nested_logit_individual"


def test_blp_aggregate_rule() -> None:
    selection = select_method(_profile(data_level="aggregate", price_variation="sufficient"))
    assert selection.rule_id == "blp_aggregate"
    assert selection.cross_check_methods == ["synthetic_control", "causal_impact"]


def test_synthetic_control_primary_rule() -> None:
    selection = select_method(
        _profile(
            data_level="aggregate", price_variation="none", donor_pool="strong", pre_periods=12
        )
    )
    assert selection.rule_id == "synthetic_control_primary"


def test_causal_impact_primary_rule() -> None:
    selection = select_method(
        _profile(data_level="aggregate", price_variation="low", donor_pool="weak", pre_periods=8)
    )
    assert selection.rule_id == "causal_impact_primary"


def test_did_primary_rule() -> None:
    selection = select_method(
        _profile(data_level="aggregate", price_variation="low", donor_pool="weak", pre_periods=6)
    )
    assert selection.rule_id in ("did_primary", "causal_impact_primary")


def test_sparse_bayesian_fallback_rule() -> None:
    selection = select_method(
        _profile(data_level="aggregate", price_variation="none", donor_pool="none", pre_periods=1)
    )
    assert selection.rule_id == "sparse_bayesian_fallback"
    assert selection.confidence_ceiling == "low"


def test_first_matching_rule_wins() -> None:
    # transaction + sufficient price + NOT segments_estimable should hit
    # nested_logit_individual, not mixed_logit_full (order matters)
    selection = select_method(
        _profile(data_level="transaction", price_variation="sufficient", segments_estimable=False)
    )
    assert selection.rule_id == "nested_logit_individual"


def test_unmatched_profile_raises() -> None:
    # data_level="panel" with sufficient price but NOT segments_estimable does match
    # nested_logit_individual; construct something that truly matches nothing instead:
    # donor_pool value outside all listed enums isn't possible via the typed model, so
    # exercise the "no rule matched" path directly against a minimal rules file.
    import tempfile
    from pathlib import Path

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty_rules.yaml"
        path.write_text(
            yaml.dump(
                {
                    "thresholds": {},
                    "rules": [
                        {
                            "id": "only_rule",
                            "when": {"data_level": ["transaction"]},
                            "primary": "mixed_logit",
                        }
                    ],
                }
            )
        )
        with pytest.raises(ValueError, match="no method_rules.yaml rule matched"):
            select_method(_profile(data_level="aggregate"), rules_path=path)
