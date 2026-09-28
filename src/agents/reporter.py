"""Reporter node (build plan section 8, node 7): narrate the validated result. This
is the one place besides `planner.py` an LLM may run, and it is read-only with respect
to numbers -- `narrate` takes an already-final `CannibalizationResult` and returns a
*new* copy with only the `narrative` field changed. No other field can be touched, so
the LLM structurally cannot alter a number (see `tests/unit/test_reporter.py`).
"""

from __future__ import annotations

from src.schema.results import CannibalizationResult

_REPORTER_SYSTEM_PROMPT = """You write a short, analyst-facing narrative for a cannibalization \
estimate that has already been computed and validated. Report the numbers given to you exactly \
as given -- never recompute, round differently, or invent a number that isn't in the input. \
Attribute drivers using the confounders_flagged and validator_flags fields (e.g. "X% of the \
incumbent's drop coincides with a confirmed competitor price cut, not the focal launch"). \
Keep it to 3-5 sentences. If confidence is "low" or the result is halted, say so plainly and why."""


def narrate(
    result: CannibalizationResult, api_key: str = "", model: str = "claude-sonnet-4-6"
) -> CannibalizationResult:
    """Returns `result` with `narrative` populated. Uses the Anthropic API when
    `api_key` is set; otherwise falls back to a deterministic template so the pipeline
    always produces a readable result offline.
    """
    if result.halted:
        narrative = (
            f"Analysis halted: {result.halt_reason}. No cannibalization estimate is "
            f"reported for {result.focal_product_id} -- see validator_flags for detail."
        )
        return result.model_copy(update={"narrative": narrative})

    narrative = _llm_narrate(result, api_key, model) if api_key else _template_narrate(result)
    return result.model_copy(update={"narrative": narrative})


def _template_narrate(result: CannibalizationResult) -> str:
    parts = [
        f"An estimated {result.overall_rate:.1%} of {result.focal_product_id}'s unit sales "
        f"came at the expense of incumbent products (95% CI: {result.ci_low:.1%} - "
        f"{result.ci_high:.1%}), via {result.method_used}"
        + (
            f" cross-checked with {', '.join(result.cross_check_methods)}"
            if result.cross_check_methods
            else ""
        )
        + f". Confidence: {result.confidence}."
    ]
    if result.segments:
        top = max(result.segments, key=lambda s: s.cannibalization_rate)
        dominant = top.dominant_source.replace("_", " ")
        parts.append(
            f"The segment most affected is {top.segment_id} at "
            f"{top.cannibalization_rate:.1%}, dominated by {dominant} substitution."
        )
    if result.confounders_flagged:
        parts.append(
            "Flagged confounders that may explain part of the movement: "
            + ", ".join(result.confounders_flagged)
            + "."
        )
    blocking = [f for f in result.validator_flags if f.severity == "blocking" and not f.passed]
    if blocking:
        parts.append(
            "Validator flags to review: "
            + "; ".join(f"{f.check} ({f.detail})" for f in blocking)
            + "."
        )
    return " ".join(parts)


def _llm_narrate(result: CannibalizationResult, api_key: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_REPORTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": result.model_dump_json()}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
