"""Schema-mapping adapter for arbitrary CSV sources. This is the proof mechanism for
industry-agnosticism: every Tier-1 adapter (`cpg_scanner`, `auto_registrations`,
`electronics_shipments`) is a thin `ColumnMapping` config on top of this one class.
Adding a new industry means writing a new mapping here, or subclassing this adapter --
never touching `src/causal/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.base_adapter import DataAdapter
from src.schema.core import AnalysisRequest, Product, SalesObservation

Granularity = str


@dataclass(frozen=True)
class ColumnMapping:
    """Maps a raw CSV's column names onto the normalized schema's fields. `attributes`
    lists extra raw columns to carry through into `Product.attributes` verbatim (the
    domain-specific feature vector used for substitutability scoring).
    """

    product_id: str
    name: str
    category: str
    brand: str
    region: str
    period: str
    units: str
    price: str
    price_tier: str | None = None
    revenue: str | None = None
    promo_flag: str | None = None
    market_size: str | None = None
    launch_date: str | None = None
    is_focal: str | None = None
    nest: str | None = None
    attributes: tuple[str, ...] = field(default_factory=tuple)
    date_format: str | None = None  # None -> let pandas infer


class GenericCSVAdapter(DataAdapter):
    """Loads a single wide/long CSV where each row is one product x region x period
    observation, mapped via `ColumnMapping`. Suitable for scanner data, registrations,
    or shipment reports alike -- the mapping, not the code, carries the domain.
    """

    source_name = "generic_csv"

    def __init__(
        self,
        csv_path: str | Path,
        mapping: ColumnMapping,
        data_level: str = "aggregate",
        focal_product_id: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.mapping = mapping
        self._data_level = data_level
        self._focal_product_id = focal_product_id
        self._df: pd.DataFrame | None = None

    def data_level(self) -> str:
        return self._data_level

    def _read(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)
        return self._df

    def _parse_period(self, raw: Any) -> date:
        m = self.mapping
        if m.date_format:
            return datetime.strptime(str(raw), m.date_format).date()
        return cast(date, pd.to_datetime(raw).date())

    def _filter_window(self, df: pd.DataFrame, request: AnalysisRequest) -> pd.DataFrame:
        m = self.mapping
        periods = pd.to_datetime(df[m.period])
        start, end = request.window
        mask = (periods.dt.date >= start) & (periods.dt.date <= end)
        if request.regions:
            mask &= df[m.region].isin(request.regions)
        return df[mask]

    def load_products(self, request: AnalysisRequest) -> list[Product]:
        m = self.mapping
        df = self._filter_window(self._read(), request)
        products: dict[str, Product] = {}
        for _, row in df.drop_duplicates(subset=[m.product_id]).iterrows():
            pid = str(row[m.product_id])
            is_focal = pid == (self._focal_product_id or request.focal_product_id)
            if m.is_focal and m.is_focal in row and pd.notna(row[m.is_focal]):
                is_focal = _to_bool(row[m.is_focal])
            launch_date = None
            if m.launch_date and pd.notna(row.get(m.launch_date)):
                launch_date = self._parse_period(row[m.launch_date])
            attributes: dict[str, float | str] = {}
            for col in m.attributes:
                if col in row and pd.notna(row[col]):
                    val = row[col]
                    attributes[col] = float(val) if _is_numeric(val) else str(val)
            products[pid] = Product(
                product_id=pid,
                name=str(row.get(m.name, pid)),
                category=str(row.get(m.category, "")),
                brand=str(row.get(m.brand, "")),
                attributes=attributes,
                price_tier=str(row.get(m.price_tier, "unknown")) if m.price_tier else "unknown",
                launch_date=launch_date,
                is_focal=is_focal,
                nest=str(row[m.nest]) if m.nest and pd.notna(row.get(m.nest)) else None,
            )
        return list(products.values())

    def load_sales(self, request: AnalysisRequest) -> list[SalesObservation]:
        m = self.mapping
        df = self._filter_window(self._read(), request)
        observations = []
        for _, row in df.iterrows():
            observations.append(
                SalesObservation(
                    product_id=str(row[m.product_id]),
                    region=str(row[m.region]),
                    period=self._parse_period(row[m.period]),
                    units=float(row[m.units]),
                    revenue=float(row[m.revenue])
                    if m.revenue and pd.notna(row.get(m.revenue))
                    else None,
                    price=float(row[m.price]),
                    promo_flag=_to_bool(row[m.promo_flag])
                    if m.promo_flag and pd.notna(row.get(m.promo_flag))
                    else False,
                    market_size=float(row[m.market_size])
                    if m.market_size and pd.notna(row.get(m.market_size))
                    else None,
                )
            )
        return observations


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)
