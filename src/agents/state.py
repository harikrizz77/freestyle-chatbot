"""The typed LangGraph state (build plan section 8). Every node reads/writes a subset
of these fields; nothing flows between nodes outside this contract. All numeric fields
are populated exclusively by `src/causal/` and `src/tools.py` calls -- see
`src/agents/validator.py` for the assertion that enforces this.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from src.agents.causal_agent import CausalAgentResult
from src.causal.nested_logit import FittedChoiceModel
from src.causal.uncertainty import MassBalanceError
from src.data.base_adapter import DataAdapter
from src.schema.core import AnalysisRequest, CustomerSegment, Product, SalesObservation
from src.schema.results import CannibalizationResult, ReducedFormResult, ValidatorFlag
from src.schema.signals import ConcurrentShockAssessment
from src.schema.tools import ChoiceSet, DataProfile, MethodSelection
from src.signals.competitor import SearchAdapter

NodeName = Literal[
    "planner",
    "data_agent",
    "relationship_agent",
    "method_selector",
    "causal_agent",
    "validator",
    "reporter",
    "halt",
]


class PipelineState(TypedDict, total=False):
    # ---- run configuration (set by the caller before invoking the graph) ----
    adapter: DataAdapter
    search_adapter: SearchAdapter | None
    competitor_brands: list[str]

    # ---- planner ----
    raw_query: str
    request: AnalysisRequest

    # ---- data_agent ----
    products: list[Product]
    sales: list[SalesObservation]
    data_profile: DataProfile
    customer_segments: list[CustomerSegment]

    # ---- relationship_agent ----
    choice_set: ChoiceSet
    candidate_incumbent_ids: list[str]

    # ---- method_selector ----
    method_selection: MethodSelection

    # ---- causal_agent ----
    causal_agent_result: CausalAgentResult
    fitted_model: FittedChoiceModel | None
    primary_result: ReducedFormResult | None
    cross_check_results: list[ReducedFormResult]
    preliminary_result: CannibalizationResult

    # ---- signals (Phase 4, optional) ----
    concurrent_shock: ConcurrentShockAssessment

    # ---- validator ----
    validator_flags: list[ValidatorFlag]
    mass_balance_error: MassBalanceError | None
    confidence: Literal["high", "medium", "low"]
    halted: bool
    halt_reason: str | None

    # ---- reporter ----
    final_result: CannibalizationResult

    # ---- control flow ----
    seed: int
    errors: list[str]
