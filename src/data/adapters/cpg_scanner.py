"""CPG scanner adapter (Dominick's Finer Foods / Nielsen-style weekly SKU-level store
sales, price, promo). Tier 1 primary validation dataset (build plan section 5): rich
price variation identifies elasticities cleanly.

This is a thin `ColumnMapping` over `GenericCSVAdapter` -- see that module's docstring
for why: the industry-agnosticism proof is that this file and `auto_registrations.py`
differ only in column names and defaults, never in causal logic.

Expected raw CSV columns (Dominick's-style long format, one row per SKU x store x
week): `upc`, `description`, `category`, `brand`, `store`, `week_ending`, `units`,
`price`, `promo`, `retail_units_all_stores` (as a market-size proxy).
"""

from __future__ import annotations

from pathlib import Path

from src.data.adapters.generic_csv import ColumnMapping, GenericCSVAdapter

DOMINICKS_MAPPING = ColumnMapping(
    product_id="upc",
    name="description",
    category="category",
    brand="brand",
    region="store",
    period="week_ending",
    units="units",
    price="price",
    revenue=None,
    promo_flag="promo",
    market_size="retail_units_all_stores",
    price_tier=None,
    launch_date="launch_date",
    is_focal="is_focal",
    nest="brand",
    attributes=("package_size", "flavor"),
)


class CPGScannerAdapter(GenericCSVAdapter):
    """Weekly SKU-level scanner panel (Dominick's / Nielsen / Circana-IRI format)."""

    source_name = "cpg_scanner"

    def __init__(
        self,
        csv_path: str | Path,
        mapping: ColumnMapping = DOMINICKS_MAPPING,
        focal_product_id: str | None = None,
    ) -> None:
        super().__init__(
            csv_path, mapping, data_level="aggregate", focal_product_id=focal_product_id
        )
