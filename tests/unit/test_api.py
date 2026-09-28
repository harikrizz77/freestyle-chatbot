from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import app

FIXTURES = Path(__file__).parent.parent / "fixtures"
client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_methods_returns_rules() -> None:
    response = client.get("/methods")
    assert response.status_code == 200
    rules = response.json()
    assert any(r["id"] == "blp_aggregate" for r in rules)


def test_analyze_end_to_end() -> None:
    payload = {
        "request": {
            "focal_product_id": "1002",
            "window": ["2024-01-01", "2024-12-31"],
        },
        "data_source": "cpg_scanner",
        "csv_path": str(FIXTURES / "cpg_scanner_sample.csv"),
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["focal_product_id"] == "1002"
    assert 0.0 <= body["overall_rate"] <= 1.0
    assert body["halted"] is False


def test_analyze_missing_csv_returns_400() -> None:
    payload = {
        "request": {"focal_product_id": "1002", "window": ["2024-01-01", "2024-12-31"]},
        "data_source": "cpg_scanner",
        "csv_path": "does/not/exist.csv",
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 400


def test_analyze_unknown_data_source_returns_422() -> None:
    payload = {
        "request": {"focal_product_id": "1002", "window": ["2024-01-01", "2024-12-31"]},
        "data_source": "not_a_real_source",
        "csv_path": str(FIXTURES / "cpg_scanner_sample.csv"),
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422  # pydantic Literal validation failure
