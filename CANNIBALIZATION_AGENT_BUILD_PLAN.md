# Cannibalization Estimation Agent — Build Specification

**Document type:** Solution architecture + implementation plan
**Intended reader:** Claude Code (autonomous build agent) and human maintainers
**Goal:** Build an industry-agnostic, multi-agent system that estimates the cannibalization rate of a newly launched product on incumbent products, accounting for both internal (price, features, sales history) and external (brand loyalty, purchasing power, competitor moves, macro/geopolitical) factors.

---

## 0. How to read this document

This is a phased spec. **Build strictly in phase order.** Do not scaffold the agent layer before the causal engine passes its acceptance tests. Each phase has explicit acceptance criteria; treat them as gates. The causal/statistical core is the source of truth — LLM agents only orchestrate, adapt data, select methods, and narrate. **No agent is ever allowed to compute a cannibalization number with an LLM.** All numbers come from deterministic, unit-tested Python.

---

## 1. Problem framing

Cannibalization is a **causal counterfactual** problem, not a correlation problem. The central quantity is:

> Of the units the new (focal) product sold, how many would have gone to the incumbent product(s) had the focal product never launched?

Everything in the system exists to estimate that counterfactual credibly. Two complementary estimation families are used:

1. **Structural discrete-choice model (primary when choice/panel or rich share+price data exist)** — nested / mixed logit. Cannibalization falls out of comparing choice probabilities in a "focal exists" world vs. a "focal removed" world.
2. **Reduced-form time-series causal methods (fallback / cross-check)** — synthetic control, difference-in-differences, Bayesian structural time series (CausalImpact). Cannibalization = incumbent actual vs. counterfactual sales, normalized by focal sales.

External factors (loyalty, purchasing power, competitor moves, macro) enter as **covariates, nest structure, and donor-selection criteria** — never as post-hoc narrative caveats.

---

## 2. Tech stack (authoritative — do not substitute without noting why)

### Language & runtime
- **Python 3.11**
- Package/dependency management: **`uv`** (fallback: Poetry). Pin all versions in `pyproject.toml`.

### Causal / statistical
- **`pyblp`** — aggregate BLP / random-coefficients & nested logit from market shares + prices (primary structural engine for aggregate data).
- **`xlogit`** — multinomial / mixed logit on individual-level (transaction/panel) data.
- **`pylogit`** — nested logit on individual-level data (used when nest structure is required at the individual level).
- **`pysyncon`** — synthetic control.
- **`tfcausalimpact`** — Bayesian structural time-series causal impact.
- **`linearmodels`** (`PanelOLS`) + **`statsmodels`** — difference-in-differences, fixed effects, diagnostics.
- **`numpy`, `pandas`, `scipy`, `scikit-learn`** — core numerics, segmentation (clustering), bootstrap.

### Agent orchestration
- **LangGraph** — the DAG orchestrator (nodes + conditional edges + typed state). Do **not** use a free-form chat/agent loop; this workflow is a graph with method-selection branching.
- **Anthropic SDK** (`anthropic`) — LLM calls for planning, data mapping, method selection, narration. Model: `claude-sonnet-4-6` for routine steps; escalate to a stronger model only for the final report if needed.
- **Pydantic v2** — the normalized schema and all tool I/O contracts. Every tool takes and returns Pydantic models.

### Data access
- **`httpx`** — HTTP client for external APIs.
- **`duckdb`** — local analytical store for scanner/panel data (fast, file-based, no server).
- **`sqlalchemy`** (optional) — if a persistent relational store is later needed.
- Web search / news for competitor signals: the Anthropic web search tool via the API, or a pluggable search adapter interface (see §7.3).
- Macro data: **FRED API** (`fredapi`), **World Bank API** (`wbgapi`).

### API / UI
- **FastAPI** — REST API exposing the pipeline.
- **Streamlit** — analyst-facing UI (estimate, confidence interval, segment breakdown, method used, confidence flags).

### Quality / ops
- **`pytest`** + **`pytest-cov`** — testing.
- **`ruff`** (lint+format), **`mypy`** (strict typing on the tools/schema layer).
- **`structlog`** — structured logging.
- **`python-dotenv`** — env/secrets loading.
- Optional observability: **LangSmith** for tracing the LangGraph runs.

---

## 3. Repository structure

```
cannibalization-agent/
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   ├── settings.py            # Pydantic Settings (env-driven config)
│   └── method_rules.yaml      # data-availability → method-selection rules
├── src/
│   ├── schema/
│   │   ├── core.py            # Product, SalesSeries, MarketContext, AnalysisRequest, etc.
│   │   ├── results.py         # CannibalizationResult, SegmentResult, ConfidenceReport
│   │   └── signals.py         # CompetitorEvent, MacroSeries, LoyaltyProfile
│   ├── data/
│   │   ├── base_adapter.py    # DataAdapter ABC → normalized schema
│   │   ├── adapters/
│   │   │   ├── cpg_scanner.py      # Dominick's / Nielsen-style scanner
│   │   │   ├── auto_registrations.py
│   │   │   ├── electronics_shipments.py
│   │   │   └── generic_csv.py      # schema-mapping adapter for arbitrary CSV
│   │   └── store.py           # DuckDB load/query helpers
│   ├── causal/
│   │   ├── nested_logit.py    # structural engine (pyblp / pylogit / xlogit wrappers)
│   │   ├── synthetic_control.py
│   │   ├── causal_impact.py
│   │   ├── did.py
│   │   ├── counterfactual.py  # two-world construction + share redistribution
│   │   └── uncertainty.py     # bootstrap / delta-method CIs, mass-balance check
│   ├── signals/
│   │   ├── segmentation.py    # loyalty / price-sensitivity / income segments
│   │   ├── competitor.py      # competitor-event sourcing → covariates/flags
│   │   └── macro.py           # FRED / World Bank → covariates
│   ├── agents/
│   │   ├── state.py           # LangGraph typed state (PipelineState)
│   │   ├── graph.py           # graph wiring + conditional edges
│   │   ├── planner.py
│   │   ├── data_agent.py
│   │   ├── relationship_agent.py  # substitutability / choice-set builder
│   │   ├── method_selector.py
│   │   ├── causal_agent.py
│   │   ├── validator.py       # HARD GATE
│   │   └── reporter.py
│   ├── api/
│   │   └── app.py             # FastAPI
│   └── ui/
│       └── dashboard.py       # Streamlit
├── tests/
│   ├── unit/                  # tool-level, deterministic, no LLM
│   ├── integration/           # graph runs on fixture data
│   └── fixtures/              # toy datasets w/ known-answer counterfactuals
└── data/
    ├── raw/                   # downloaded source data (gitignored)
    └── processed/             # normalized parquet/duckdb
```

---

## 4. Normalized data schema (the contract)

Every adapter maps raw data into these Pydantic models. **All causal and agent code operates only on these** — industry specifics live exclusively in adapters.

```python
# schema/core.py  (illustrative field set — implement as Pydantic v2)


class Product:
    product_id: str
    name: str
    category: str
    brand: str
    attributes: dict[str, float | str]  # feature vector (spec/feature fit)
    price_tier: str  # e.g. entry/mid/premium
    launch_date: date | None
    is_focal: bool  # the new launch under study
    nest: str | None  # brand/segment nest label for nested logit


class SalesObservation:
    product_id: str
    region: str
    period: date  # weekly/monthly granularity
    units: float
    revenue: float | None
    price: float
    promo_flag: bool
    market_size: float | None  # for share computation


class MarketContext:
    region: str
    period: date
    macro_index: dict[str, float]  # inflation, FX, disposable income, etc.
    competitor_pressure: dict[str, float]  # per rival: price/availability shock


class CustomerSegment:  # only when panel/transaction data exists
    segment_id: str
    loyalty_score: float  # 0..1 per brand
    price_sensitivity: float
    income_proxy: float
    size: float


class AnalysisRequest:
    focal_product_id: str
    candidate_incumbents: list[str] | None
    window: tuple[date, date]
    regions: list[str] | None
    granularity: Literal["week", "month"]


# schema/results.py
class SegmentResult:
    segment_id: str
    cannibalization_rate: float
    ci_low: float
    ci_high: float
    dominant_source: str  # e.g. "within-brand" vs "competitor" vs "new demand"


class CannibalizationResult:
    focal_product_id: str
    overall_rate: float
    ci_low: float
    ci_high: float
    segments: list[SegmentResult]
    method_used: str
    confidence: Literal["high", "medium", "low"]
    confounders_flagged: list[str]
    narrative: str
```

---

## 5. Data sources (concrete, tiered by reliability)

Build and validate on Tier 1 first. Each source gets its own adapter conforming to `DataAdapter`.

**Tier 1 (prototype + validate here):**
- **CPG scanner — Dominick's Finer Foods (Chicago Booth Kilts Center, free).** Weekly SKU-level store sales, price, promo. *Primary validation dataset.* Rich price variation → identifies elasticities cleanly.
- **CPG panel — Nielsen/NIQ or Circana/IRI** (if licensed). Household-level choice data → enables true segmentation and individual-level mixed logit.
- **Automotive registrations** — SIAM (India), WardsAuto/Experian (US). Trim/model/region, public, long price history, clean launch windows.
- **Consumer electronics shipments** — IDC / Counterpoint / Canalys / CIRP (aggregate; matches the iPhone demo). Expect aggregate share+price, so lean on `pyblp` + synthetic control, not individual choice.
- **Subscription/SaaS tier data** (internal, if available) — cleanest possible self-analysis case; customer-level upgrade/downgrade logs.

**Tier 2 (workable, noisier):** apparel (Circana/NPD), pharma/OTC (IQVIA — heavy regulatory confounds), airlines (OAG + gov stats — dynamic pricing confound).

**Tier 3 (weak public data — require proprietary access):** B2B/industrial equipment, real estate/large durables, luxury goods.

**External-signal sources:**
- Macro: **FRED** (inflation, FX, disposable income), **World Bank** (cross-country purchasing power).
- Competitor moves: news/web search adapter + any available pricing feeds.

---

## 6. Causal engine (Phase 1 — build first, no LLMs)

### 6.1 Structural discrete-choice model
- Implement aggregate nested logit / random-coefficients logit via **`pyblp`** for share+price data.
- Implement individual-level nested/mixed logit via **`pylogit`/`xlogit`** for panel/transaction data.
- Utility spec (per customer/segment *i*, product *j*, time *t*):
  ```
  U = β0_j + β_loyalty·Loyalty(i,brand_j) + β_price·Price(j,t)/Income(i)
      + β_features·FeatureMatch(i,j) + β_comp·CompetitorPressure(j,t)
      + β_macro·MacroIndex(region,t) + ε
  ```
- **Nest structure** encodes brand loyalty via nesting parameter λ (within-brand substitution vs. cross-brand). Choice set must include a **no-purchase / keep-current** option.

### 6.2 Counterfactual computation (`counterfactual.py`)
- Construct two worlds with all covariates frozen: **Factual** (focal present) and **Counterfactual** (focal utility → −∞ / removed).
- Recompute choice probabilities; redistribute focal's mass — within-nest first (governed by λ), then across nests to competitors / no-purchase.
- Aggregate:
  ```
  cannibalized_units(incumbent j') = Σ_i [P_CF(i,j') − P_F(i,j')] · MarketSize(i)
  cannibalization_rate = cannibalized_units(j') / total_focal_units
  ```
- Compute **per segment**, then aggregate. Report per-segment dominant source (within-brand / competitor / net-new demand).

### 6.3 Reduced-form cross-checks
- **Synthetic control** (`pysyncon`): build donor pool of unaffected products/regions matched on pre-period trajectory **and** loyalty/purchasing-power/competitor-exposure profile.
- **CausalImpact** (`tfcausalimpact`): forecast incumbent from pre-period + covariates; gap = impact.
- **DiD** (`linearmodels.PanelOLS`): incumbent vs. matched controls, external factors as explicit control variables; test parallel pre-trends.

### 6.4 Uncertainty (`uncertainty.py`)
- Propagate coefficient uncertainty via **bootstrap** (resample coefficient vector) or **delta method** → distribution of cannibalization rates. Report point estimate + CI, never a bare number.
- **Mass-balance check:** removed focal demand must (approximately) redistribute, not vanish or double-count. Failing this = implementation bug → block.

### 6.5 Phase 1 acceptance criteria
- Nested logit fits Dominick's data; holdout hit-rate / log-likelihood reported.
- Counterfactual runs end-to-end producing a rate + bootstrap CI on a toy fixture with a **known analytic answer** (`tests/fixtures`).
- Structural estimate and at least one reduced-form estimate agree within a documented tolerance on ≥1 real launch; disagreements investigated and explained.
- Mass-balance test passes. All causal functions have unit tests. `mypy` clean on `causal/`.

---

## 7. Tools layer (Phase 2 — wrap causal core, still no agent reasoning)

Expose each capability as a typed, deterministic, independently unit-tested function taking/returning Pydantic models:

- `load_data(source, request) -> list[SalesObservation]`
- `build_choice_set(products, request) -> ChoiceSet`  (substitutability via attribute-embedding similarity — domain-agnostic)
- `fit_structural_model(data, choice_set) -> FittedModel`
- `run_synthetic_control(...) -> ReducedFormResult`
- `run_causal_impact(...) -> ReducedFormResult`
- `run_did(...) -> ReducedFormResult`
- `compute_counterfactual(fitted_model) -> CannibalizationResult`
- `bootstrap_ci(...) -> ConfidenceReport`
- `segment_customers(panel) -> list[CustomerSegment]`
- `fetch_competitor_events(window, category, region) -> list[CompetitorEvent]`
- `fetch_macro(region, window) -> MacroSeries`

### 7.3 Adapter & search interfaces
- `DataAdapter` ABC: `.load() -> normalized schema`. New industry = new adapter, **zero changes to causal code** (this is the agnosticism proof).
- `SearchAdapter` ABC for competitor signals, so web-search backend is swappable.

**Phase 2 acceptance:** every tool has unit tests passing without any LLM call; adapters for ≥2 industries (CPG + one other) both produce valid normalized data.

---

## 8. Agent layer (Phase 3 — LangGraph)

Typed `PipelineState` flows through nodes. LLM does **only**: query parsing, messy-data→schema mapping, method selection from a checklist, narration.

**Nodes & responsibilities:**
1. **Planner** — parse `AnalysisRequest` from NL query; identify focal + candidate incumbents + window.
2. **Data agent** — pick adapter, load, map to schema, report data granularity/coverage.
3. **Relationship agent** — build choice set / substitutability scores (embedding similarity over attributes).
4. **Method selector** — apply `config/method_rules.yaml`: e.g. *panel + price variation → mixed logit; aggregate shares + price variation → pyblp nested logit; only time series + good donor pool → synthetic control/CausalImpact; sparse data → Bayesian-shrinkage lightweight logit.* Conditional LangGraph edge routes to the chosen causal path.
5. **Causal agent** — run selected method + ≥1 cross-check via the tools; produce `CannibalizationResult`.
6. **Validator (HARD GATE)** — checks: pre-trend parallelism (DiD) / pre-period fit (SC/CausalImpact) / IIA test (logit); mass-balance; concurrent competitor or macro shock in window. If any fails → set `confidence="low"` and attach flags, or halt with a clear reason. **Never emit a confident number over a failed gate.**
7. **Reporter** — narrate result with CI, segment breakdown, and attribution ("X% of incumbent drop explained by competitor price cut, not focal launch").

**Conditional edges:** method_selector → {structural | reduced-form} paths; validator → {reporter | low-confidence reporter | halt}.

**Phase 3 acceptance:** full graph runs on fixture data start→finish; validator provably blocks a seeded bad case (injected concurrent competitor launch); LLM never produces the numeric estimate (assert numbers originate from tool outputs).

---

## 9. External signals (Phase 4)

- **Segmentation** (`segmentation.py`): with panel data, cluster on loyalty (repeat-purchase/tenure), price sensitivity, income proxy; OR use mixed-logit random coefficients to estimate β distributions without hard buckets. Without panel data, fall back to region / price-tier segments.
- **Competitor agent**: web/news search for rival launches, price cuts, stock-outs in the window → covariates (`CompetitorPressure`) and validator flags.
- **Macro agent**: FRED + World Bank series → `MacroIndex` covariates (FX, inflation, disposable income) and region-time controls.
- Feed all of these **into** donor selection / covariate sets / utility terms — re-estimate, don't annotate.

**Phase 4 acceptance:** competitor & macro covariates measurably change the estimate on a case where an overlapping event exists; segment-level rates produced on panel data.

---

## 10. API & UI (Phase 5)

- **FastAPI**: `POST /analyze` (AnalysisRequest → CannibalizationResult), `GET /methods`, `GET /health`. Async; validates via Pydantic.
- **Streamlit**: input focal product + window + data source; display point estimate, CI band, per-segment breakdown, method used, confidence badge, flagged confounders, and the narrative.

---

## 11. Cross-cutting requirements

- **Config/secrets:** all keys (Anthropic, FRED) via `.env` + Pydantic Settings; provide `.env.example`. Never hardcode.
- **Logging:** `structlog` JSON logs; log method chosen, data coverage, validator outcomes, CIs.
- **Determinism:** fix random seeds for bootstrap/estimation; record seed in results for reproducibility.
- **Typing:** `mypy --strict` on `schema/`, `causal/`, tool layer.
- **Testing:** unit (tools, deterministic), integration (graph on fixtures), and at least one known-answer counterfactual fixture. Target ≥80% coverage on `causal/` and `schema/`.
- **Error handling:** insufficient-data and estimation-failure paths must return a structured low-confidence result, never a crash or a hallucinated number.
- **Extensibility rule (enforced by review):** adding an industry must touch only `data/adapters/` and possibly `method_rules.yaml`. Any change to `causal/` to support a new industry is a design violation.

---

## 12. Build order & milestones (recap)

| Phase | Deliverable | Gate |
|------|-------------|------|
| 1 | Causal engine (nested logit + counterfactual + ≥1 reduced-form + CI + mass-balance), no LLM | Known-answer fixture passes; two methods agree on 1 real launch |
| 2 | Typed tool layer + ≥2 industry adapters, unit-tested | All tools pass without LLM |
| 3 | LangGraph agent graph + hard-gate validator | Graph runs end-to-end; validator blocks seeded bad case |
| 4 | Segmentation + competitor + macro signals as covariates | Overlapping-event case changes estimate; segment rates produced |
| 5 | FastAPI + Streamlit | Analyze-endpoint + dashboard render estimate, CI, segments, confidence |

**Golden rule:** prove industry-agnosticism by adding the *second* industry adapter (e.g. auto registrations) in Phase 2 without editing any `causal/` code. That is the architecture's key demonstration.

---

## 13. Explicit non-goals (for v1)

- Real-time streaming inference.
- Fully automated data-source discovery (adapters are curated).
- Causal claims on Tier 3 industries without proprietary data.
- Any LLM-generated numeric estimate — permanently out of scope by design.
