"""Macro covariates (build plan sections 7, 9): FRED (US) + World Bank (cross-country
purchasing power) -> `MacroSeries`, fed into the utility spec's `MacroIndex(region,t)`
term and DiD/CausalImpact control variables.

Like `competitor.SearchAdapter`, the data source is pluggable via `MacroSource` so
`fetch_macro` itself stays a pure, unit-testable function -- network calls and API keys
live only inside the concrete `FredMacroSource` / `WorldBankMacroSource` classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from src.schema.signals import MacroSeries

# Default FRED series for common indicators; extend as needed -- adding a region or
# indicator here never requires touching src/causal/.
DEFAULT_FRED_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",
    "disposable_income": "DSPIC96",
    "fx_broad_dollar_index": "DTWEXBGS",
}

# Default World Bank indicator codes for cross-country purchasing power.
DEFAULT_WORLD_BANK_INDICATORS: dict[str, str] = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "inflation": "FP.CPI.TOTL.ZG",
}


class MacroSource(ABC):
    """Swappable macro-data backend."""

    @abstractmethod
    def fetch(self, indicator: str, region: str, window: tuple[date, date]) -> dict[date, float]:
        """Return {period: value} for `indicator` in `region` within `window`."""


class StaticMacroSource(MacroSource):
    """Test/offline adapter: returns caller-supplied series."""

    def __init__(self, series: dict[tuple[str, str], dict[date, float]]) -> None:
        self._series = series

    def fetch(self, indicator: str, region: str, window: tuple[date, date]) -> dict[date, float]:
        start, end = window
        raw = self._series.get((indicator, region), {})
        return {d: v for d, v in raw.items() if start <= d <= end}


class FredMacroSource(MacroSource):
    """US macro indicators via FRED. Requires `settings.fred_api_key`. Lazy-imports
    `fredapi` so this module has no hard dependency on the `signals` extra.
    """

    def __init__(self, api_key: str, series_map: dict[str, str] | None = None) -> None:
        self._api_key = api_key
        self._series_map = series_map or DEFAULT_FRED_SERIES

    def fetch(self, indicator: str, region: str, window: tuple[date, date]) -> dict[date, float]:
        from fredapi import Fred

        series_id = self._series_map.get(indicator)
        if series_id is None:
            raise ValueError(f"no FRED series mapped for indicator {indicator!r}")
        fred = Fred(api_key=self._api_key)
        series = fred.get_series(series_id, observation_start=window[0], observation_end=window[1])
        return {ts.date(): float(v) for ts, v in series.items() if v == v}  # drop NaN


class WorldBankMacroSource(MacroSource):
    """Cross-country purchasing power via the World Bank API. `region` is an ISO3
    country code. Lazy-imports `wbgapi` so this module has no hard dependency on the
    `signals` extra.
    """

    def __init__(self, indicator_map: dict[str, str] | None = None) -> None:
        self._indicator_map = indicator_map or DEFAULT_WORLD_BANK_INDICATORS

    def fetch(self, indicator: str, region: str, window: tuple[date, date]) -> dict[date, float]:
        import wbgapi as wb

        code = self._indicator_map.get(indicator)
        if code is None:
            raise ValueError(f"no World Bank indicator mapped for {indicator!r}")
        years = range(window[0].year, window[1].year + 1)
        df = wb.data.DataFrame(code, economy=region, time=years)
        result: dict[date, float] = {}
        for col in df.columns:
            year = int(str(col).replace("YR", ""))
            value = df[col].iloc[0]
            if value == value:  # not NaN
                result[date(year, 1, 1)] = float(value)
        return result


def fetch_macro(
    source: MacroSource, region: str, window: tuple[date, date], indicators: list[str]
) -> list[MacroSeries]:
    """Pure orchestration: one `MacroSeries` per indicator, sourced via `source`."""
    return [
        MacroSeries(
            region=region,
            indicator=indicator,
            series=source.fetch(indicator, region, window),
            source=type(source).__name__,
        )
        for indicator in indicators
    ]
