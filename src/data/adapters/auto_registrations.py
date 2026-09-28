"""Automotive registrations adapter (SIAM India / WardsAuto-Experian US style). Tier 1
(build plan section 5): trim/model/region, public, long price history, clean launch
windows.

Deliberately a *different* raw schema from `cpg_scanner.py` (monthly registrations by
model/trim/region rather than weekly SKU/store scanner rows) to demonstrate that adding
an industry only requires a new `ColumnMapping` -- see `generic_csv.py` and build plan
section 12's "golden rule" (add the second adapter without touching `src/causal/`).

Expected raw CSV columns: `model_trim_code`, `model_name`, `segment`, `manufacturer`,
`region`, `month`, `registrations`, `avg_transaction_price`, `total_segment_registrations`.
"""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.generic_csv import ColumnMapping, GenericCSVAdapter

AUTO_REGISTRATIONS_MAPPING = ColumnMapping(
    product_id="model_trim_code",
    name="model_name",
    category="segment",
    brand="manufacturer",
    region="region",
    period="month",
    units="registrations",
    price="avg_transaction_price",
    revenue=None,
    promo_flag=None,
    market_size="total_segment_registrations",
    price_tier="price_tier",
    launch_date="launch_date",
    is_focal="is_focal",
    nest="manufacturer",
    attributes=("body_type", "fuel_type", "engine_displacement_l"),
)


class AutoRegistrationsAdapter(GenericCSVAdapter):
    """Monthly model/trim registrations panel (SIAM / WardsAuto-Experian format)."""

    source_name = "auto_registrations"

    def __init__(
        self,
        csv_path: str | Path,
        mapping: ColumnMapping = AUTO_REGISTRATIONS_MAPPING,
        focal_product_id: str | None = None,
    ) -> None:
        super().__init__(
            csv_path, mapping, data_level="aggregate", focal_product_id=focal_product_id
        )
