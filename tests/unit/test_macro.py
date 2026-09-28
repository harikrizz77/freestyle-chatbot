from datetime import date

from src.signals.macro import StaticMacroSource, fetch_macro


def test_fetch_macro_filters_window_and_labels_source() -> None:
    series = {
        ("cpi", "US"): {
            date(2023, 12, 1): 300.0,
            date(2024, 1, 1): 301.0,
            date(2024, 6, 1): 305.0,
        }
    }
    source = StaticMacroSource(series)
    result = fetch_macro(source, "US", (date(2024, 1, 1), date(2024, 3, 1)), ["cpi"])
    assert len(result) == 1
    assert result[0].indicator == "cpi"
    assert result[0].region == "US"
    assert result[0].series == {date(2024, 1, 1): 301.0}
    assert result[0].source == "StaticMacroSource"


def test_fetch_macro_missing_series_returns_empty() -> None:
    source = StaticMacroSource({})
    result = fetch_macro(source, "US", (date(2024, 1, 1), date(2024, 3, 1)), ["cpi"])
    assert result[0].series == {}
