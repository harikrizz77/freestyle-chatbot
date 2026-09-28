"""Validator node (build plan section 8, node 6) -- the HARD GATE. Never emits a
confident number over a failed check: applies `method_rules.yaml`'s `overrides`
section, plus the pre-trend/pre-period-fit/mass-balance checks build plan 6.5 and 8
require. This is the one place confidence is decided; `causal_agent.py`'s
preliminary confidence value is always provisional and gets overwritten here.
"""

from __future__ import annotations

from src.agents.causal_agent import CausalAgentResult
from src.causal.uncertainty import MassBalanceError, assert_mass_balance, cross_check_disagreement
from src.schema.results import CannibalizationResult, Confidence, ValidatorFlag
from src.schema.signals import ConcurrentShockAssessment
from src.schema.tools import MethodSelection

_CONFIDENCE_ORDER: dict[Confidence, int] = {"high": 2, "medium": 1, "low": 0}
_PRE_PERIOD_FIT_MIN = 0.5


def _cap(confidence: Confidence, ceiling: Confidence) -> Confidence:
    return confidence if _CONFIDENCE_ORDER[confidence] <= _CONFIDENCE_ORDER[ceiling] else ceiling


def validate(
    causal_result: CausalAgentResult,
    method_selection: MethodSelection,
    concurrent_shock: ConcurrentShockAssessment,
    cross_check_tolerance: float = 0.10,
) -> CannibalizationResult:
    prelim = causal_result.preliminary_result
    flags: list[ValidatorFlag] = []
    confidence: Confidence = "high"
    confounders: list[str] = []

    # 1. Mass balance -- an implementation bug, not a data problem. Halt immediately.
    if prelim.mass_balance is not None:
        try:
            assert_mass_balance(prelim.mass_balance)
            flags.append(ValidatorFlag(check="mass_balance", passed=True, severity="info"))
        except MassBalanceError as exc:
            flags.append(
                ValidatorFlag(
                    check="mass_balance", passed=False, detail=str(exc), severity="blocking"
                )
            )
            return prelim.model_copy(
                update={
                    "confidence": "low",
                    "validator_flags": flags,
                    "halted": True,
                    "halt_reason": f"mass balance check failed: {exc}",
                }
            )

    # 2. Pre-period fit (synthetic control / causal impact cross-checks and primary).
    for result in [*causal_result.primary_reduced_form, *causal_result.cross_check_reduced_form]:
        if result.pre_period_fit_score is not None:
            passed = result.pre_period_fit_score >= _PRE_PERIOD_FIT_MIN
            flags.append(
                ValidatorFlag(
                    check=f"pre_period_fit[{result.method}:{result.incumbent_product_id}]",
                    passed=passed,
                    detail=f"fit_score={result.pre_period_fit_score:.3f}",
                    severity="warning",
                )
            )
            if not passed:
                confidence = _cap(confidence, "medium")

    # 3. Parallel pre-trends (DiD). A real IIA-style structural-model check would
    # require refitting on a restricted choice set (Hausman-McFadden); recording the
    # nesting parameters as a transparency note in place of that full test for v1.
    for result in [*causal_result.primary_reduced_form, *causal_result.cross_check_reduced_form]:
        if result.parallel_trends_passed is not None:
            flags.append(
                ValidatorFlag(
                    check=f"parallel_trends[{result.incumbent_product_id}]",
                    passed=result.parallel_trends_passed,
                    severity="blocking" if not result.parallel_trends_passed else "info",
                )
            )
            if not result.parallel_trends_passed:
                confidence = "low"

    if causal_result.fitted_model is not None:
        lambdas = causal_result.fitted_model.lambdas
        flags.append(
            ValidatorFlag(
                check="iia_note",
                passed=True,
                detail=f"nesting lambdas={lambdas} (a value near 1 approaches plain MNL/IIA)",
                severity="info",
            )
        )

    # 4. Cross-check agreement (method_rules.yaml cross_check_agreement_tolerance).
    if causal_result.cross_check_overall_rate is not None:
        cross_check_rate = causal_result.cross_check_overall_rate
        disagree = cross_check_disagreement(
            prelim.overall_rate, cross_check_rate, focal_units=1.0, tolerance=cross_check_tolerance
        )
        flags.append(
            ValidatorFlag(
                check="cross_check_agreement",
                passed=not disagree,
                detail=f"primary={prelim.overall_rate:.4f} cross_check={cross_check_rate:.4f}",
                severity="warning" if disagree else "info",
            )
        )
        if disagree:
            confidence = _cap(confidence, "medium")

    # 5. Concurrent competitor/macro shock override.
    if concurrent_shock.level == "confirmed":
        if not concurrent_shock.covariate_included:
            flags.append(
                ValidatorFlag(
                    check="concurrent_shock",
                    passed=False,
                    detail=concurrent_shock.rationale,
                    severity="blocking",
                )
            )
            confounders.append("confirmed_competitor_or_macro_shock_not_modeled")
            return prelim.model_copy(
                update={
                    "confidence": "low",
                    "validator_flags": flags,
                    "confounders_flagged": confounders,
                    "halted": True,
                    "halt_reason": (
                        "a confirmed concurrent competitor/macro shock overlaps the analysis "
                        "window and was not modeled as a covariate -- the estimate would "
                        "conflate the focal launch's effect with this confound"
                    ),
                }
            )
        confidence = "low"
        confounders.append("confirmed_competitor_or_macro_shock")
        flags.append(
            ValidatorFlag(
                check="concurrent_shock",
                passed=True,
                detail=concurrent_shock.rationale,
                severity="warning",
            )
        )
    elif concurrent_shock.level == "suspected":
        confounders.append("suspected_competitor_or_macro_shock")
        confidence = _cap(confidence, "medium")
        flags.append(
            ValidatorFlag(
                check="concurrent_shock",
                passed=True,
                detail=concurrent_shock.rationale,
                severity="info",
            )
        )

    # 6. The rule's own confidence ceiling (e.g. sparse_bayesian_fallback -> low).
    if method_selection.confidence_ceiling is not None:
        confidence = _cap(confidence, method_selection.confidence_ceiling)

    return prelim.model_copy(
        update={
            "confidence": confidence,
            "validator_flags": flags,
            "confounders_flagged": confounders,
            "halted": False,
            "halt_reason": None,
        }
    )
