# Cannibalization Estimation Agent

Industry-agnostic, multi-agent system that estimates the cannibalization rate of a
newly launched product on incumbent products, accounting for internal factors (price,
features, sales history) and external factors (brand loyalty, purchasing power,
competitor moves, macro shocks).

Cannibalization is treated as a **causal counterfactual** problem, not a correlation
problem: *of the units the focal product sold, how many would have gone to an
incumbent had the focal product never launched?* Every number in the output traces to
a deterministic, unit-tested Python function in `src/causal/` -- **no LLM ever computes
a cannibalization number**. LLMs (when configured) only parse queries, select methods
from a fixed rule table, and narrate an already-validated result.

See `CANNIBALIZATION_AGENT_BUILD_PLAN.md` for the full specification this was built
against; this README documents what's actually implemented against that spec.

## Status: all 5 phases implemented

| Phase | What | Gate |
|---|---|---|
| 1 | Causal engine: native nested-logit MLE + counterfactual + bootstrap/delta-method CI + mass-balance check + synthetic control / CausalImpact / DiD cross-checks | `tests/unit/test_counterfactual.py` reproduces a hand-derived known answer; mass balance holds to float precision |
| 2 | Typed tools layer + 3 industry adapters (CPG scanner, auto registrations, electronics shipments) over one `GenericCSVAdapter` | `tests/integration/test_industry_agnosticism.py` runs two different industries through the same `build_choice_set` call, zero branching |
| 3 | LangGraph agent graph (planner → data_agent → relationship_agent → competitor_signal → macro_signal → method_selector → causal_agent → validator → reporter) with a hard-gate validator | `tests/unit/test_validator.py` proves a confirmed, unmodeled shock still halts; `tests/integration/test_agent_graph.py` proves the graph runs end-to-end |
| 4 | Segmentation (KMeans / region-price-tier fallback), competitor event sourcing, FRED/World Bank macro adapters -- **folded into the fitted model as covariates**, not just used to flag a confound | `tests/unit/test_tools.py::test_build_aggregate_design_folds_in_competitor_pressure`; `tests/integration/test_agent_graph.py` proves a confirmed shock that's modeled downgrades confidence instead of halting, and that supplying competitor/macro data measurably changes the estimate |
| 5 | FastAPI (`/analyze`, `/methods`, `/health`) + Streamlit dashboard | `tests/unit/test_api.py` |

**80/80 tests passing**, `ruff check .` clean, `mypy` clean on `src/schema`, `src/causal`,
`src/data` (strict mode). Coverage on `src/causal` and `src/schema` (the spec's explicit
target) is 90%+; see `pytest --cov=src --cov-report=term-missing`.

## Quickstart

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # core + test/lint/type tooling
pytest                             # 49 tests, ~1 minute

# optional extras, install as needed:
uv pip install -e ".[structural]"    # pyblp / xlogit / pylogit (see note below)
uv pip install -e ".[reduced-form]"  # pysyncon / tfcausalimpact / linearmodels
uv pip install -e ".[agents]"        # langgraph / anthropic / langsmith
uv pip install -e ".[signals]"       # fredapi / wbgapi
uv pip install -e ".[api]"           # fastapi / uvicorn
uv pip install -e ".[ui]"            # streamlit
```

Run the API:
```bash
uvicorn src.api.app:app --reload
```

Run the dashboard:
```bash
streamlit run src/ui/dashboard.py
```

Try it against the bundled fixture data (no external data source needed):
```bash
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d '{
  "request": {"focal_product_id": "1002", "window": ["2024-01-01", "2024-12-31"]},
  "data_source": "cpg_scanner",
  "csv_path": "tests/fixtures/cpg_scanner_sample.csv"
}'
```

## Architecture

```
config/            method_rules.yaml (data-availability -> method rules), settings.py
src/schema/        Pydantic v2 contract: core.py, results.py, signals.py, tools.py
src/causal/        nested_logit, counterfactual, uncertainty, synthetic_control,
                    causal_impact, did -- the only place a number is computed
src/data/          DataAdapter ABC + GenericCSVAdapter + 3 industry adapters + DuckDB store
src/tools.py        typed functions wrapping the causal core (build_choice_set,
                    build_aggregate_design, build_reduced_form_series, fit_structural_model)
src/signals/        segmentation, competitor event sourcing, FRED/World Bank macro
src/agents/         LangGraph state + nodes: planner, data_agent, relationship_agent,
                    method_selector, causal_agent, validator (hard gate), reporter
src/api/            FastAPI app
src/ui/             Streamlit dashboard
tests/fixtures/     hand-derived known-answer nested-logit fixture + simulated MLE
                    recovery fixtures + sample CSVs for two different industries
```

### Design decisions worth knowing about (deviations/extensions from the spec)

- **One native structural engine, two data shapes.** `pyblp`/`pylogit`/`xlogit` wrappers
  exist (`src/causal/nested_logit.py`) but their long-format wiring is adapter-specific
  and left as a documented extension point (`fit_via_*` raise `NotImplementedError` and
  `src/tools.py:fit_structural_model` falls back to the native estimator with a logged
  warning). The native nested-logit MLE (`fit_nested_logit`) supports **both**
  individual-level choice data (`choice_index`, one chosen alternative per row) and
  aggregate/grouped share data (`choice_weights`, observed unit counts per market) via
  the same log-likelihood, which is the standard grouped-multinomial-logit identity --
  this is what lets `mixed_logit`, `nested_logit_individual`, and `blp_nested_logit`
  (three different `method_rules.yaml` rule names for three different data regimes)
  all route through one tested, dependency-light implementation.
- **Reduced-form cross-checks are pure numpy/scipy**, not `pysyncon`/`tfcausalimpact`/
  `linearmodels`, so `src/causal/` is fully unit-testable without the heavy optional
  `reduced-form` extra (which pulls in TensorFlow). They implement the identical
  estimators (Abadie et al. convex donor weighting; OLS-forecast counterfactual with an
  analytic prediction interval; two-way fixed-effects DiD) -- see each module's
  docstring for the "primary vs. fallback" framing, same as the pyblp/pylogit/xlogit
  integration point above.
- **Synthetic-control / CausalImpact confidence intervals are a documented
  approximation** (residual-variance propagation), not a full in-space placebo/
  permutation test -- called out as a natural v2 improvement in the module docstrings.
- **Reduced-form donor pools default to "the other incumbents in the choice set"**
  when no explicit donor list is supplied (`src/agents/causal_agent.py`). A production
  deployment should pass products verified unaffected by the focal launch instead;
  this is flagged in that module's docstring.
- Real Dominick's/Nielsen panel data isn't fetchable from this sandbox (licensed/
  network-gated), so the Phase 1 "holdout hit-rate" acceptance criterion is validated
  against data **simulated from the model itself with known ground-truth parameters**
  (`tests/fixtures/simulated_transactions.py`) -- this exercises the identical
  estimation machinery a real panel would, and the MLE recovers the generating
  parameters within tolerance (`tests/unit/test_fit_nested_logit.py`).

### Extensibility proof (build plan section 12's "golden rule")

Adding the auto-registrations adapter after the CPG scanner adapter touched only
`src/data/adapters/auto_registrations.py` -- zero changes to `src/causal/`. Both
adapters are ~40-line `ColumnMapping` configs over `GenericCSVAdapter`; see
`tests/integration/test_industry_agnosticism.py`.

## Testing

```bash
pytest                                          # everything
pytest tests/unit/test_counterfactual.py        # the Phase 1 hard gate
pytest tests/integration                        # cross-industry + full graph
ruff check . && ruff format --check .
mypy                                             # strict on schema/, causal/, data/
```
