from datetime import date

import pytest
from pydantic import ValidationError

from src.schema.core import AnalysisRequest, SalesObservation
from src.schema.results import ConfidenceReport, MassBalanceCheck


def test_analysis_request_rejects_backwards_window() -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(
            focal_product_id="A1",
            window=(date(2024, 6, 1), date(2024, 1, 1)),
        )


def test_analysis_request_accepts_ordered_window() -> None:
    req = AnalysisRequest(
        focal_product_id="A1",
        window=(date(2024, 1, 1), date(2024, 6, 1)),
    )
    assert req.granularity == "week"


def test_sales_observation_rejects_units_exceeding_market_size() -> None:
    with pytest.raises(ValidationError):
        SalesObservation(
            product_id="A1",
            region="US",
            period=date(2024, 1, 1),
            units=1000,
            price=10.0,
            market_size=500,
        )


def test_confidence_report_rejects_point_outside_ci() -> None:
    with pytest.raises(ValidationError):
        ConfidenceReport(
            point_estimate=1.5,
            ci_low=0.0,
            ci_high=1.0,
            method="bootstrap",
            seed=42,
        )


def test_mass_balance_check_passed_property() -> None:
    ok = MassBalanceCheck(focal_units_removed=100.0, redistributed_units_total=100.0000001)
    assert ok.passed
    bad = MassBalanceCheck(focal_units_removed=100.0, redistributed_units_total=90.0)
    assert not bad.passed
