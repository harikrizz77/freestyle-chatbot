from datetime import date

from src.signals.competitor import (
    SearchResult,
    StaticSearchAdapter,
    assess_concurrent_shock,
    fetch_competitor_events,
)


def _results() -> list[SearchResult]:
    return [
        SearchResult(
            title="RivalCo launches new zesty chip line",
            snippet="RivalCo today announced the launch of a new product",
            url="http://example.com/1",
            published=date(2024, 2, 5),
            source="news",
        ),
        SearchResult(
            title="Quarterly earnings call transcript",
            snippet="Nothing product related here",
            url="http://example.com/2",
            published=date(2024, 2, 10),
            source="news",
        ),
    ]


def test_fetch_competitor_events_classifies_launch() -> None:
    adapter = StaticSearchAdapter(_results())
    events = fetch_competitor_events(
        adapter,
        window=(date(2024, 1, 1), date(2024, 3, 1)),
        category="salty_snacks",
        region="US",
        competitor_brands=["RivalCo"],
    )
    assert len(events) == 2
    launch_events = [e for e in events if e.kind == "competitor_launch"]
    assert len(launch_events) == 1
    assert launch_events[0].confidence > 0.5


def test_fetch_competitor_events_respects_window() -> None:
    adapter = StaticSearchAdapter(_results())
    events = fetch_competitor_events(
        adapter,
        window=(date(2024, 2, 6), date(2024, 3, 1)),
        category="salty_snacks",
        region="US",
        competitor_brands=["RivalCo"],
    )
    assert len(events) == 1  # the launch event (2024-02-05) is now outside the window


def test_assess_concurrent_shock_confirmed_on_material_high_confidence_event() -> None:
    adapter = StaticSearchAdapter(_results())
    events = fetch_competitor_events(
        adapter,
        window=(date(2024, 1, 1), date(2024, 3, 1)),
        category="salty_snacks",
        region="US",
        competitor_brands=["RivalCo"],
    )
    assessment = assess_concurrent_shock(events, covariate_included=True)
    assert assessment.level == "confirmed"


def test_assess_concurrent_shock_none_when_no_events() -> None:
    assessment = assess_concurrent_shock([], covariate_included=False)
    assert assessment.level == "none"
