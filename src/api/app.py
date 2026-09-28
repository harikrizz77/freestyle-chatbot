"""FastAPI layer (build plan section 10): `POST /analyze`, `GET /methods`,
`GET /health`. Thin -- all it does is pick an adapter and hand off to
`src.agents.graph.run_pipeline_sync`; no causal logic lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from config.settings import METHOD_RULES_PATH
from src.agents.graph import run_pipeline_sync
from src.data.adapters.auto_registrations import AutoRegistrationsAdapter
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.data.adapters.electronics_shipments import ElectronicsShipmentsAdapter
from src.data.base_adapter import DataAdapter
from src.schema.core import AnalysisRequest
from src.schema.results import CannibalizationResult

app = FastAPI(
    title="Cannibalization Estimation Agent",
    description="Industry-agnostic causal estimation of new-product cannibalization.",
    version="0.1.0",
)

DataSource = Literal["cpg_scanner", "auto_registrations", "electronics_shipments"]

_ADAPTER_FACTORIES: dict[DataSource, type[DataAdapter]] = {
    "cpg_scanner": CPGScannerAdapter,
    "auto_registrations": AutoRegistrationsAdapter,
    "electronics_shipments": ElectronicsShipmentsAdapter,
}


class AnalyzeAPIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: AnalysisRequest
    data_source: DataSource
    csv_path: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/methods")
async def methods() -> list[dict]:
    """Returns the method-selection rules (`config/method_rules.yaml`) verbatim, for
    analyst transparency into what the pipeline could choose and why."""
    with open(METHOD_RULES_PATH) as f:
        config = yaml.safe_load(f)
    return config["rules"]


@app.post("/analyze", response_model=CannibalizationResult)
async def analyze(payload: AnalyzeAPIRequest) -> CannibalizationResult:
    adapter_cls = _ADAPTER_FACTORIES.get(payload.data_source)
    if adapter_cls is None:
        raise HTTPException(status_code=400, detail=f"unknown data_source: {payload.data_source}")

    csv_path = Path(payload.csv_path)
    if not csv_path.exists():
        raise HTTPException(status_code=400, detail=f"csv_path not found: {csv_path}")

    adapter = adapter_cls(csv_path)
    try:
        result = await run_in_threadpool(run_pipeline_sync, adapter, payload.request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result
