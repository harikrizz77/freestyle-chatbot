"""Reporter node (build plan section 8, node 7): narrate the validated result. This
is the one place besides `planner.py` an LLM may run, and it is read-only with respect
to numbers -- `narrate` takes an already-final `CannibalizationResult` and returns a
*new* copy with only the `narrative` field changed. No other field can be touched, so
the LLM structurally cannot alter a number (see `tests/unit/test_reporter.py`).

The narrative is written for a non-technical reader: it explains, in plain language,
what data went in, why a particular method was chosen, how the "two worlds" (with the
new product vs. without it) comparison actually produces a number, and what the
confidence level and any flags mean -- not just the bottom-line percentage. This is
the project's transparency requirement: nobody should have to trust the number
without being told how it was reached.
"""

from __future__ import annotations

from src.schema.results import CannibalizationResult
from src.schema.tools import DataLevel, DataProfile, MethodName, MethodSelection

_REPORTER_SYSTEM_PROMPT = """You write an analyst-facing explanation of a cannibalization
estimate that has already been computed and validated by deterministic code -- you are
explaining a finished result, not producing one. Report every number exactly as given to
you -- never recompute, round differently, estimate, or invent a number that isn't in the
input.

Your reader may have no statistics background, so write in plain, everyday language and
avoid jargon without explaining it first. Be thorough and transparent rather than terse:
walk the reader through
(1) what question is being answered, in one plain sentence,
(2) what data was used and roughly how much, drawing on the data_profile you're given,
(3) why this particular method was chosen over the alternatives, drawing on the
    method_selection's notes and rule_id,
(4) how the method actually works, in an intuitive way (e.g. for a choice model, explain
    comparing a "with the new product" world to a "without it" world and seeing where
    customers' demand goes instead; for a time-series method, explain forecasting what
    would have happened and comparing that to what did happen),
(5) what the resulting number and confidence interval mean in practical terms,
(6) what the segment breakdown and dominant_source values mean, if present,
(7) any confounders_flagged or validator_flags, explained as plain caveats,
(8) a one-line bottom-line summary.

Use clear section headings (Markdown ##) for readability. If confidence is "low" or the
result is halted, say so plainly, early, and explain in simple terms why the result
can't be fully trusted."""


_DATA_LEVEL_EXPLANATIONS: dict[DataLevel, str] = {
    "transaction": (
        "individual customer purchase records -- we could see, for each purchase, "
        "exactly which product a specific customer chose"
    ),
    "panel": (
        "repeat-purchase household data -- we could track the same customers' choices "
        "over time, though not quite at the level of every single transaction"
    ),
    "aggregate": (
        "store- and time-period-level sales totals -- we could see how many units of "
        "each product sold in total, but not which individual customer bought what"
    ),
}

_METHOD_EXPLANATIONS: dict[MethodName, str] = {
    "mixed_logit": (
        "We built a mathematical model of how individual customers choose between "
        "products, allowing different customers to weigh price, brand loyalty, and "
        "features differently. This is the most detailed approach and lets us "
        "estimate cannibalization separately for different types of customers, not "
        "just one overall number."
    ),
    "nested_logit_individual": (
        "We built a mathematical model of how individual customers choose between "
        "products, grouping similar products together (e.g. products from the same "
        "brand) since customers tend to treat products within a group as closer "
        "substitutes for one another than products from a different group."
    ),
    "blp_nested_logit": (
        "We built a mathematical model of customer choice from store-level sales and "
        "price data (rather than individual purchase records), using a well-"
        "established technique from economics for exactly this situation -- "
        "inferring how customers choose between products when you can only see total "
        "sales, not individual choices."
    ),
    "synthetic_control": (
        "We built a 'synthetic twin' of each competing product out of a weighted "
        "blend of other, unaffected products, chosen so the blend closely matched "
        "that competitor's sales pattern *before* the new product launched. After "
        "launch, we compared what the competitor actually sold to what its synthetic "
        "twin sold -- the gap is our estimate of the impact."
    ),
    "causal_impact": (
        "We used each competing product's own sales history before the launch to "
        "forecast what it likely would have kept selling afterward if nothing had "
        "changed, then compared that forecast to what it actually sold. The "
        "difference is attributed to the new product's launch."
    ),
    "did": (
        "We compared the change in sales for the affected product against the change "
        "in sales for similar, unaffected products over the same period (a "
        "'difference-in-differences' comparison) -- this cancels out broader trends "
        "that would have affected everyone anyway, isolating the effect of the new "
        "launch specifically."
    ),
    "bayesian_shrinkage_logit": (
        "There wasn't enough data here to fit a full statistical model with real "
        "confidence, so we produced a directional, back-of-envelope estimate "
        "instead, based on how similar the new product is to each competitor. This "
        "is a starting-point estimate, not a rigorous one -- treat it as a rough "
        "signal rather than a firm number."
    ),
}

_DOMINANT_SOURCE_EXPLANATIONS: dict[str, str] = {
    "within_brand": (
        "most of the affected customers switched to a *different product from the "
        "same brand* -- the company selling the new product mostly took sales from "
        "itself, not from rivals"
    ),
    "competitor": (
        "most of the affected customers switched to a *rival brand's* product -- the "
        "new product mostly won customers away from competitors, not from its own "
        "sibling products"
    ),
    "net_new_demand": (
        "most of this segment's demand appears to be genuinely new -- customers who "
        "would not have bought anything at all if the new product hadn't launched"
    ),
    "unknown": "there wasn't a clear majority pattern one way or the other here",
}

_PRICE_VARIATION_EXPLANATIONS = {
    "sufficient": ("enough movement over time to reliably tell how price-sensitive customers are"),
    "low": "only a little movement, so price sensitivity is a weaker part of the picture",
    "none": "essentially no movement, so price wasn't a useful signal here",
}

_CONFIDENCE_EXPLANATIONS = {
    "high": (
        "**High** confidence means the data was rich enough, the checks all passed, "
        "and an independent cross-check method agreed closely with the main "
        "estimate. You can treat this number as reliable for decision-making, while "
        "still keeping the confidence interval in mind."
    ),
    "medium": (
        "**Medium** confidence means the estimate is usable but came with some "
        "yellow flags -- for example, a cross-check method didn't fully agree, or "
        "the historical data used for comparison wasn't a perfect fit. Treat the "
        "number as a reasonable planning estimate, not a precise fact."
    ),
    "low": (
        "**Low** confidence means real limitations were found -- sparse data, a "
        "failed statistical check, or an outside event overlapping the launch (even "
        "one we did model as a covariate still means the estimate is entangled with "
        "that event, not a clean read on the launch alone). Use this only as a rough "
        "directional signal, and see the flags "
        "below for exactly why."
    ),
}


def narrate(
    result: CannibalizationResult,
    api_key: str = "",
    model: str = "claude-sonnet-4-6",
    data_profile: DataProfile | None = None,
    method_selection: MethodSelection | None = None,
) -> CannibalizationResult:
    """Returns `result` with `narrative` populated. Uses the Anthropic API when
    `api_key` is set; otherwise falls back to a deterministic template so the
    pipeline always produces a readable, transparent explanation offline.
    `data_profile` and `method_selection` are optional context used to explain
    *why* this method was chosen and what data supported it -- pass them when
    available (the graph always has both by the time this node runs; ad-hoc
    callers may omit them).
    """
    if result.halted:
        narrative = _halted_narrative(result, method_selection)
        return result.model_copy(update={"narrative": narrative})

    if api_key:
        narrative = _llm_narrate(result, api_key, model, data_profile, method_selection)
    else:
        narrative = _template_narrate(result, data_profile, method_selection)
    return result.model_copy(update={"narrative": narrative})


def _halted_narrative(
    result: CannibalizationResult, method_selection: MethodSelection | None
) -> str:
    lines = [
        "## The short version",
        (
            f"We stopped short of giving a cannibalization estimate for "
            f"**{result.focal_product_id}** because something in the data made the "
            "calculation untrustworthy -- we'd rather tell you that plainly than "
            "hand you a number we don't believe ourselves."
        ),
        "",
        "## Why we stopped",
        result.halt_reason or "An internal safety check failed; see the flags below.",
        "",
        "## What was flagged",
    ]
    for flag in result.validator_flags:
        status = "OK" if flag.passed else "FAILED"
        suffix = f" -- {flag.detail}" if flag.detail else ""
        lines.append(f"- **{flag.check}** [{status}]{suffix}")

    if method_selection is not None:
        would_use = (
            f"Based on the data available, the method we would have used is "
            f"**{method_selection.primary_method}**"
        )
        if method_selection.notes:
            would_use += f" ({method_selection.notes})"
        would_use += "."
        lines += ["", "## What we would have used", would_use]

    lines += [
        "",
        "## What you can do about it",
        (
            "Either supply more/cleaner data for this launch, or explicitly account "
            "for the flagged issue (e.g. include the competitor event as a modeled "
            "factor) and re-run the analysis."
        ),
    ]
    return "\n".join(lines)


def _template_narrate(
    result: CannibalizationResult,
    data_profile: DataProfile | None,
    method_selection: MethodSelection | None,
) -> str:
    lines: list[str] = []
    lines += _bottom_line_section(result)
    lines += _question_section(result)
    if data_profile is not None:
        lines += _data_section(data_profile)
    lines += _factors_section(result)
    lines += _method_section(result, method_selection)
    if result.segments:
        lines += _segments_section(result)
    if result.mass_balance is not None:
        lines += _mass_balance_section(result)
    has_flags_worth_mentioning = result.confounders_flagged or any(
        f.severity in ("warning", "blocking") for f in result.validator_flags
    )
    if has_flags_worth_mentioning:
        lines += _caveats_section(result)
    lines += [
        "## How much should you trust this?",
        _CONFIDENCE_EXPLANATIONS.get(result.confidence, ""),
    ]
    return "\n".join(lines)


def _bottom_line_section(result: CannibalizationResult) -> list[str]:
    return [
        "## Bottom line",
        (
            f"An estimated **{result.overall_rate:.1%}** of {result.focal_product_id}'s "
            "sales came at the expense of existing products, rather than being sales "
            "that wouldn't have happened otherwise. We're 95% confident the true "
            f"figure is somewhere between **{result.ci_low:.1%} and "
            f"{result.ci_high:.1%}**."
        ),
        "",
    ]


def _question_section(result: CannibalizationResult) -> list[str]:
    return [
        "## The question we're answering",
        (
            f"When {result.focal_product_id} launched, did it mostly create "
            "brand-new sales, or did it mostly just take sales away from products "
            "that already existed? We answer this by comparing two hypothetical "
            "worlds: one where the new product exists (what actually happened), and "
            "one where it never launched at all. The difference between those two "
            "worlds -- how much *more* the older products would have sold if the "
            'new one never existed -- is what we call "cannibalization."'
        ),
        "",
    ]


def _data_section(data_profile: DataProfile) -> list[str]:
    level_desc = _DATA_LEVEL_EXPLANATIONS.get(data_profile.data_level, data_profile.data_level)
    coverage = data_profile.coverage_notes or "the supplied data"
    price_desc = _PRICE_VARIATION_EXPLANATIONS.get(
        data_profile.price_variation, data_profile.price_variation
    )
    summary = (
        f"We worked with {level_desc}. In total: {coverage}. Prices in this data "
        f"showed {price_desc}, and we had {data_profile.pre_periods} time period(s) "
        "of history from before the launch to establish a 'normal' baseline."
    )
    return ["## What data we used", summary, ""]


def _factors_section(result: CannibalizationResult) -> list[str]:
    """Answers, for this specific run, "was it only sales data, or were competitor
    moves/macro conditions factored in too?" -- explicit and unconditional, not just
    surfaced when something went wrong, per `src/agents/validator.py`'s
    `external_factors_included` flag.
    """
    flag = next((f for f in result.validator_flags if f.check == "external_factors_included"), None)
    if flag is None:
        return []
    return ["## What factors fed into this number", flag.detail, ""]


def _method_section(
    result: CannibalizationResult, method_selection: MethodSelection | None
) -> list[str]:
    lines = ["## Why we used this method, and how it works"]
    if method_selection is not None and method_selection.notes:
        lines.append(
            "Given the shape of the available data, the system's rule-based method "
            f"selector chose this approach: *{method_selection.notes}*"
        )
    lines.append(_METHOD_EXPLANATIONS.get(result.method_used, f"We used `{result.method_used}`."))
    if result.cross_check_methods:
        cross_checks = "; ".join(f"**{m}**" for m in result.cross_check_methods)
        lines.append(
            "As a sanity check, we also ran this through one or more independent "
            f"methods ({cross_checks}) to see if they told a similar story. "
            "Agreement between independent methods is one of the strongest signals "
            "that a result can be trusted; disagreement lowers our confidence (see "
            "below)."
        )
    lines.append("")
    return lines


def _segments_section(result: CannibalizationResult) -> list[str]:
    lines = [
        "## Breaking it down further",
        (
            "Rather than one single number, we also calculated cannibalization "
            "separately for each of the following groupings (e.g. store/time-"
            "period, or customer segment), because the effect isn't always the "
            "same everywhere:"
        ),
    ]
    for seg in result.segments:
        dominant = _DOMINANT_SOURCE_EXPLANATIONS.get(seg.dominant_source, seg.dominant_source)
        rate = f"{seg.cannibalization_rate:.1%}"
        lines.append(f"- **{seg.segment_id}**: {rate} cannibalization -- {dominant}.")
    lines.append("")
    return lines


def _mass_balance_section(result: CannibalizationResult) -> list[str]:
    assert result.mass_balance is not None
    verdict = (
        "That check passed: the math accounts for all the units."
        if result.mass_balance.passed
        else (
            "That check FAILED, which means there's a bug or data issue "
            "undermining this result -- treat this specific run's numbers with "
            "real skepticism."
        )
    )
    explanation = (
        "Before trusting this number, the system checked that every unit of the "
        "new product's sales we say was 'taken from somewhere' actually landed "
        "somewhere -- either with a specific competing product or as genuinely new "
        f"demand -- rather than mysteriously vanishing or being double-counted. {verdict}"
    )
    return ["## A built-in sanity check", explanation, ""]


def _caveats_section(result: CannibalizationResult) -> list[str]:
    lines = ["## Things to watch out for"]
    if result.confounders_flagged:
        confounders = ", ".join(c.replace("_", " ") for c in result.confounders_flagged)
        lines.append(
            "Other things happening around the same time may have influenced these "
            f"numbers alongside the launch itself: {confounders}."
        )
    interesting_flags = [f for f in result.validator_flags if f.severity in ("warning", "blocking")]
    for flag in interesting_flags:
        suffix = f": {flag.detail}" if flag.detail else ""
        lines.append(f"- **{flag.check}**{suffix}")
    lines.append("")
    return lines


def _llm_narrate(
    result: CannibalizationResult,
    api_key: str,
    model: str,
    data_profile: DataProfile | None,
    method_selection: MethodSelection | None,
) -> str:
    import json

    import anthropic

    payload = {
        "result": json.loads(result.model_dump_json()),
        "data_profile": json.loads(data_profile.model_dump_json()) if data_profile else None,
        "method_selection": (
            json.loads(method_selection.model_dump_json()) if method_selection else None
        ),
    }

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=_REPORTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
