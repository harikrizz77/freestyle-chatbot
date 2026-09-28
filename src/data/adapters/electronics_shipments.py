"""Consumer electronics shipments adapter (IDC / Counterpoint / Canalys / CIRP style).
Tier 1 (build plan section 5) -- matches the canonical "iPhone launch cannibalizes
prior-gen iPhone" demo. Aggregate share+price data, so the method selector should route
this to `blp_nested_logit` + synthetic control / CausalImpact cross-checks rather than
an individual-level structural model.

Expected raw CSV columns: `sku_code`, `model_name`, `form_factor`, `oem`, `region`,
`quarter`, `shipment_units`, `asp_usd`, `total_market_shipment_units`.
"""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.generic_csv import ColumnMapping, GenericCSVAdapter

ELECTRONICS_SHIPMENTS_MAPPING = ColumnMapping(
    product_id="sku_code",
    name="model_name",
    category="form_factor",
    brand="oem",
    region="region",
    period="quarter",
    units="shipment_units",
    price="asp_usd",
    revenue=None,
    promo_flag=None,
    market_size="total_market_shipment_units",
    price_tier="price_tier",
    launch_date="launch_date",
    is_focal="is_focal",
    nest="oem",
    attributes=("storage_gb", "screen_size_in", "chipset_tier"),
)


class ElectronicsShipmentsAdapter(GenericCSVAdapter):
    """Quarterly aggregate shipment panel (IDC / Counterpoint / Canalys / CIRP format)."""

    source_name = "electronics_shipments"

    def __init__(
        self,
        csv_path: str | Path,
        mapping: ColumnMapping = ELECTRONICS_SHIPMENTS_MAPPING,
        focal_product_id: str | None = None,
    ) -> None:
        super().__init__(
            csv_path, mapping, data_level="aggregate", focal_product_id=focal_product_id
        )
