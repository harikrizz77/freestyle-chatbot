"""Competitor-event sourcing (build plan sections 7.3, 9): news/web search -> structured
`CompetitorEvent` covariates and validator flags. The search backend is pluggable via
`SearchAdapter` so it can be swapped (Anthropic web search tool, a pricing feed, a news
API) without touching how events become covariates.

Classification here is a deterministic keyword classifier by default -- swappable for
an LLM-based classifier in the agent layer (`src/agents/`) since classifying a news
snippet's *type* is NL mapping, not a numeric estimate, and is explicitly allowed to be
LLM-assisted per the build plan's division of labor (section 8). Whatever classifies
events, `fetch_competitor_events` and `assess_concurrent_shock` are pure functions of
their input so they stay unit-testable without network access or an API key.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from src.schema.signals import CompetitorEvent, ConcurrentShockAssessment, EventKind

_KEYWORD_RULES: tuple[tuple[EventKind, tuple[str, ...]], ...] = (
    ("competitor_launch", ("launch", "unveil", "debut", "introduce")),
    ("price_cut", ("price cut", "slashes price", "discount", "reduces price")),
    ("price_increase", ("price hike", "raises price", "price increase")),
    ("stockout", ("stockout", "out of stock", "supply shortage", "sold out")),
    ("promotion", ("promotion", "sale event", "limited-time offer")),
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    snippet: str
    url: str
    published: date
    source: str


class SearchAdapter(ABC):
    """Swappable web-search backend."""

    @abstractmethod
    def search(self, query: str, window: tuple[date, date]) -> list[SearchResult]:
        """Return raw search results for `query` published within `window`."""


class StaticSearchAdapter(SearchAdapter):
    """Test/offline adapter: returns a fixed, caller-supplied result list. Also useful
    for wiring in results fetched out-of-band (e.g. a pre-downloaded news export).
    """

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, window: tuple[date, date]) -> list[SearchResult]:
        start, end = window
        return [r for r in self._results if start <= r.published <= end]


class AnthropicWebSearchAdapter(SearchAdapter):
    """Production adapter: the Anthropic API's web search tool. Requires
    `settings.anthropic_api_key`. Import of `anthropic` is lazy so this module has no
    hard dependency on the `agents` extra.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def search(self, query: str, window: tuple[date, date]) -> list[SearchResult]:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": f"{query} between {window[0]} and {window[1]}"}],
        )
        results: list[SearchResult] = []
        for block in response.content:
            if getattr(block, "type", None) == "web_search_tool_result":
                for item in getattr(block, "content", []):
                    results.append(
                        SearchResult(
                            title=getattr(item, "title", ""),
                            snippet=getattr(item, "encrypted_content", "") or "",
                            url=getattr(item, "url", ""),
                            published=window[1],  # API doesn't always return a date
                            source="anthropic_web_search",
                        )
                    )
        return results


def _classify(text: str) -> EventKind:
    lowered = text.lower()
    for kind, keywords in _KEYWORD_RULES:
        if any(re.search(re.escape(kw), lowered) for kw in keywords):
            return kind
    return "other"


def fetch_competitor_events(
    search_adapter: SearchAdapter,
    window: tuple[date, date],
    category: str,
    region: str,
    competitor_brands: list[str],
) -> list[CompetitorEvent]:
    """One query per competitor brand, classified via `_classify`, returned as
    normalized `CompetitorEvent`s ready to feed into covariates / donor selection.
    """
    events: list[CompetitorEvent] = []
    for brand in competitor_brands:
        query = f"{brand} {category} {region} price launch stockout"
        for i, result in enumerate(search_adapter.search(query, window)):
            kind = _classify(f"{result.title} {result.snippet}")
            events.append(
                CompetitorEvent(
                    event_id=f"{brand}_{result.published.isoformat()}_{i}",
                    competitor_brand=brand,
                    category=category,
                    region=region,
                    event_date=result.published,
                    kind=kind,
                    source=result.source,
                    confidence=0.6 if kind != "other" else 0.3,
                    summary=result.title,
                )
            )
    return events


def assess_concurrent_shock(
    events: list[CompetitorEvent], covariate_included: bool, min_confidence: float = 0.5
) -> ConcurrentShockAssessment:
    """Rolls sourced events up into the validator's `concurrent_shock` verdict
    (`method_rules.yaml` overrides section): confirmed if a high-confidence,
    materially disruptive event overlaps the window; suspected if a lower-confidence
    or ambiguous one does; none otherwise.
    """
    material_kinds = {"competitor_launch", "price_cut", "stockout"}
    material = [e for e in events if e.kind in material_kinds]
    high_confidence = [e for e in material if e.confidence >= min_confidence]

    if high_confidence:
        level = "confirmed"
        rationale = (
            f"{len(high_confidence)} high-confidence competitor event(s) "
            f"({', '.join(sorted({e.kind for e in high_confidence}))}) overlap the window."
        )
    elif material:
        level = "suspected"
        rationale = f"{len(material)} low-confidence competitor event(s) overlap the window."
    else:
        level = "none"
        rationale = "No material competitor events found in the window."

    return ConcurrentShockAssessment(
        level=level, events=events, covariate_included=covariate_included, rationale=rationale
    )
