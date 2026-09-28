"""Build plan section 12's "golden rule": two different industries (CPG scanner,
auto registrations) must both produce valid normalized data and flow through the
*exact same* `build_choice_set` tool with zero branching on source type.
"""

from datetime import date
from pathlib import Path

import pytest

from src.data.adapters.auto_registrations import AutoRegistrationsAdapter
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.schema.core import AnalysisRequest
from src.tools import build_choice_set

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.parametrize(
    ("adapter_cls", "csv_name", "focal_id"),
    [
        (CPGScannerAdapter, "cpg_scanner_sample.csv", "1002"),
        (AutoRegistrationsAdapter, "auto_registrations_sample.csv", "X100-EV"),
    ],
)
def test_adapter_produces_valid_choice_set(adapter_cls, csv_name, focal_id) -> None:
    adapter = adapter_cls(FIXTURES / csv_name)
    request = AnalysisRequest(
        focal_product_id=focal_id,
        window=(date(2024, 1, 1), date(2024, 12, 31)),
    )
    products = adapter.load_products(request)
    sales = adapter.load_sales(request)

    assert len(products) >= 2
    assert len(sales) > 0
    assert any(p.is_focal for p in products)

    choice_set = build_choice_set(products, request)
    assert choice_set.focal_product_id == focal_id
    assert any(e.product_id == "__no_purchase__" for e in choice_set.entries)
    assert len(choice_set.candidate_incumbent_ids) >= 1
    for entry in choice_set.entries:
        assert 0.0 <= entry.substitutability_to_focal <= 1.0
