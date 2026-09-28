"""Planner node (build plan section 8, node 1): parse an `AnalysisRequest` from a
natural-language query. The LLM's job here is query parsing/slot-filling only -- it
never sees or produces a cannibalization number. `parse_request` degrades gracefully
to a regex/heuristic parse when no Anthropic API key is configured, so the graph
remains runnable (with a less flexible planner) in offline/test environments.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

from src.schema.core import AnalysisRequest

_PLANNER_SYSTEM_PROMPT = """You turn an analyst's natural-language question into a strict \
JSON object matching this schema (no extra keys, no prose):
{"focal_product_id": str, "candidate_incumbents": list[str] | null, \
"window_start": "YYYY-MM-DD", "window_end": "YYYY-MM-DD", \
"regions": list[str] | null, "granularity": "week" | "month"}
The focal_product_id is the newly launched product being analyzed for cannibalization. \
If a date range isn't stated, infer a reasonable default window from any launch date \
mentioned, defaulting to +/- 6 months around it. Respond with ONLY the JSON object."""


def parse_request(
    query: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-6",
    default_focal_product_id: str | None = None,
) -> AnalysisRequest:
    """Parse `query` into an `AnalysisRequest`. Uses the Anthropic API when `api_key`
    is set; otherwise falls back to `_heuristic_parse` (regex-based, best-effort).
    """
    if api_key:
        payload = _llm_parse(query, api_key, model)
    else:
        payload = _heuristic_parse(query, default_focal_product_id)

    return AnalysisRequest(
        focal_product_id=payload["focal_product_id"],
        candidate_incumbents=payload.get("candidate_incumbents"),
        window=(
            date.fromisoformat(payload["window_start"]),
            date.fromisoformat(payload["window_end"]),
        ),
        regions=payload.get("regions"),
        granularity=payload.get("granularity", "week"),
    )


def _llm_parse(query: str, api_key: str, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_PLANNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    return json.loads(text)


def _heuristic_parse(query: str, default_focal_product_id: str | None) -> dict:
    """Best-effort regex parse for offline/test use. Looks for an explicit product id
    token and an explicit date range; otherwise falls back to a 12-month window ending
    today. This is deliberately conservative -- it under-parses rather than guesses a
    business-meaningful focal product id from free text.
    """
    focal_match = re.search(
        r"\bfocal[_ ]?(?:product)?[:\s]+([A-Za-z0-9_\-]+)", query, re.IGNORECASE
    )
    focal_id = focal_match.group(1) if focal_match else default_focal_product_id
    if focal_id is None:
        raise ValueError(
            "could not determine focal_product_id from query without an LLM; "
            "pass default_focal_product_id explicitly or configure ANTHROPIC_API_KEY"
        )

    date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", query)
    if len(date_matches) >= 2:
        window_start, window_end = sorted(date_matches[:2])
    elif len(date_matches) == 1:
        anchor = date.fromisoformat(date_matches[0])
        window_start = (anchor - timedelta(days=180)).isoformat()
        window_end = (anchor + timedelta(days=180)).isoformat()
    else:
        today = date.today()
        window_start = (today - timedelta(days=365)).isoformat()
        window_end = today.isoformat()

    granularity = "month" if "month" in query.lower() else "week"

    return {
        "focal_product_id": focal_id,
        "candidate_incumbents": None,
        "window_start": window_start,
        "window_end": window_end,
        "regions": None,
        "granularity": granularity,
    }
