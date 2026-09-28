"""Streamlit analyst dashboard (build plan section 10). Run with:

    streamlit run src/ui/dashboard.py

Calls `run_pipeline_sync` directly (no API server required) -- the UI is a thin
presentation layer over the same pipeline `src/api/app.py` exposes over HTTP.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.graph import run_pipeline_sync
from src.data.adapters.auto_registrations import AutoRegistrationsAdapter
from src.data.adapters.cpg_scanner import CPGScannerAdapter
from src.data.adapters.electronics_shipments import ElectronicsShipmentsAdapter
from src.schema.core import AnalysisRequest

st.set_page_config(page_title="Cannibalization Estimator", layout="wide")

_ADAPTERS = {
    "CPG scanner (Dominick's / Nielsen-style)": CPGScannerAdapter,
    "Auto registrations (SIAM / WardsAuto-style)": AutoRegistrationsAdapter,
    "Electronics shipments (IDC / Counterpoint-style)": ElectronicsShipmentsAdapter,
}

_CONFIDENCE_COLOR = {"high": "green", "medium": "orange", "low": "red"}

st.title("Cannibalization Estimation Agent")
st.caption(
    "Causal, not correlational: every number below comes from a fitted structural "
    "choice model or a reduced-form counterfactual -- never from an LLM."
)

with st.sidebar:
    st.header("Analysis inputs")
    data_source_label = st.selectbox("Data source", list(_ADAPTERS.keys()))
    csv_path = st.text_input("CSV path", placeholder="tests/fixtures/cpg_scanner_sample.csv")
    focal_product_id = st.text_input("Focal product id")
    window_start = st.date_input("Window start", value=date(2024, 1, 1))
    window_end = st.date_input("Window end", value=date(2024, 12, 31))
    run_clicked = st.button("Run analysis", type="primary")

if run_clicked:
    if not csv_path or not Path(csv_path).exists():
        st.error(f"CSV path not found: {csv_path!r}")
    elif not focal_product_id:
        st.error("Enter a focal product id.")
    else:
        adapter_cls = _ADAPTERS[data_source_label]
        adapter = adapter_cls(csv_path)
        request = AnalysisRequest(
            focal_product_id=focal_product_id, window=(window_start, window_end)
        )
        with st.spinner("Running causal pipeline..."):
            try:
                result = run_pipeline_sync(adapter, request)
            except Exception as exc:  # surfaced to the analyst, not swallowed
                st.exception(exc)
                result = None

        if result is not None:
            if result.halted:
                st.error(f"Analysis halted: {result.halt_reason}")
            else:
                badge = _CONFIDENCE_COLOR.get(result.confidence, "gray")
                col1, col2, col3 = st.columns(3)
                col1.metric("Cannibalization rate", f"{result.overall_rate:.1%}")
                col2.metric("95% CI", f"{result.ci_low:.1%} - {result.ci_high:.1%}")
                col3.markdown(f"**Confidence:** :{badge}[{result.confidence.upper()}]")

                st.subheader("Method")
                st.write(
                    f"**Primary:** {result.method_used}"
                    + (
                        f"  |  **Cross-checked with:** {', '.join(result.cross_check_methods)}"
                        if result.cross_check_methods
                        else ""
                    )
                )

                if result.segments:
                    st.subheader("Segment breakdown")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "segment": s.segment_id,
                                    "cannibalization_rate": s.cannibalization_rate,
                                    "ci_low": s.ci_low,
                                    "ci_high": s.ci_high,
                                    "dominant_source": s.dominant_source,
                                }
                                for s in result.segments
                            ]
                        ),
                        use_container_width=True,
                    )

                if result.confounders_flagged:
                    st.subheader("Flagged confounders")
                    for c in result.confounders_flagged:
                        st.warning(c)

                st.subheader("Narrative")
                st.write(result.narrative)

                with st.expander("Validator flags"):
                    st.dataframe(
                        pd.DataFrame([f.model_dump() for f in result.validator_flags]),
                        use_container_width=True,
                    )
else:
    st.info("Configure inputs in the sidebar and click **Run analysis**.")
