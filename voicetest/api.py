"""
api.py — FastAPI application for voicetest.

Endpoints
---------
POST /runs               Create a run and start the background job.
GET  /runs/{id}          Run status + progress + per-call summaries.
GET  /runs/{id}/report   Aggregate report (pass rate + per-call breakdown).
GET  /scenarios          List all builtin scenarios.
GET  /recordings/{call_id}  Serve the recording WAV for a call.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from voicetest.agent_model.extractor import AgentModelExtractor, make_extractor
from voicetest.agent_model.models import AgentModel
from voicetest.config import Settings
from voicetest.jobs import get_builtin_scenarios, start_battery, start_run
from voicetest.providers.telephony import AllowlistError, check_allowlist
from voicetest.scenarios.compiler import ScenarioCompiler, make_compiler
from voicetest.scenarios.loader import load_all
from voicetest.store import RunStatus, Store

app = FastAPI(title="voicetest API", version="0.1.0")

# ---------------------------------------------------------------------------
# Shared state (one store instance, lazily created from env)
# ---------------------------------------------------------------------------

_store: Store | None = None
_settings: Settings | None = None
_compiler: ScenarioCompiler | None = None
_extractor: AgentModelExtractor | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def _get_compiler() -> ScenarioCompiler:
    global _compiler
    if _compiler is None:
        _compiler = make_compiler()
    return _compiler


def _get_extractor() -> AgentModelExtractor:
    global _extractor
    if _extractor is None:
        _extractor = make_extractor()
    return _extractor


# ---------------------------------------------------------------------------
# Request / response shapes
# ---------------------------------------------------------------------------


class CompileRequest(BaseModel):
    description: str


class CompiledScenarioItem(BaseModel):
    id: str
    title: str
    category: str
    difficulty: str
    compiler_version: int | None


class CompileResponse(BaseModel):
    scenarios: list[CompiledScenarioItem]
    # Full scenario dicts so the UI can immediately start a run
    scenario_dicts: list[dict[str, Any]]


class CreateRunRequest(BaseModel):
    scenario_id: str
    number: str
    n: int


class CreateBatteryRequest(BaseModel):
    description: str
    number: str = "loopback"
    known_issues: str = ""
    # Optional: if provided, only run these scenario IDs (from a prior /battery/preview call)
    selected_scenario_ids: list[str] | None = None
    # Optional: id of a confirmed AgentModel (from /agent-model/{id}/confirm).
    # When set, scenario generation is grounded in it and the report includes
    # per-node graph coverage.
    agent_model_id: str | None = None
    # Optional: auto-add the last 5 production-failure-derived regression
    # scenarios (tagged group="regression") to this battery's batch.
    include_production_regressions: bool = False


class PreviewBatteryRequest(BaseModel):
    description: str
    known_issues: str = ""
    # Optional: id of a confirmed AgentModel — grounds generation in it.
    agent_model_id: str | None = None


class BatteryPreviewScenario(BaseModel):
    id: str
    title: str
    group: str
    category: str
    node_id: str = ""


class BatteryPreviewResponse(BaseModel):
    scenarios: list[BatteryPreviewScenario]


class GenerateForNodeRequest(BaseModel):
    agent_model_id: str
    node_id: str
    count: int = 10


class GenerateRegressionRequest(BaseModel):
    agent_model_id: str | None = None
    root_cause_description: str


class ProductionCallIn(BaseModel):
    transcript: str
    outcome: str
    error: str | None = None


class IngestProductionCallsRequest(BaseModel):
    agent_model_id: str | None = None
    calls: list[ProductionCallIn]


class IngestProductionCallsResponse(BaseModel):
    new_failures: int
    regression_scenarios_generated: int


class ProductionFailureItem(BaseModel):
    id: str
    agent_id: str
    transcript: str
    outcome: str
    error: str | None
    root_cause: str
    detected_at: str


class ProductionFailuresResponse(BaseModel):
    failures: list[ProductionFailureItem]


class GenerateScenarioFromFailureRequest(BaseModel):
    call_id: str


class GroupProgress(BaseModel):
    total: int
    passed: int
    failed: int
    running: int


class BatteryStatusResponse(BaseModel):
    battery_id: str
    description: str
    status: str
    total: int
    completed: int
    groups: dict[str, GroupProgress]


class BatteryReportResponse(BaseModel):
    battery_id: str
    status: str
    report: dict[str, Any] | None


class CallSummary(BaseModel):
    call_id: str
    call_index: int
    outcome: str | None
    duration_ms: int | None
    passed: bool | None
    summary: str | None
    error: str | None


class RunStatusResponse(BaseModel):
    run_id: str
    scenario_id: str
    number: str
    n: int
    status: str
    completed: int
    calls: list[CallSummary]
    error: str | None = None


class RunReportResponse(BaseModel):
    run_id: str
    scenario_id: str
    total: int
    completed: int
    pass_count: int
    pass_rate: float
    calls: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_confirmed_agent_model(agent_model_id: str | None) -> AgentModel | None:
    """
    Load a confirmed AgentModel by id, or None if not provided.

    Raises 404 if the id doesn't resolve and 422 if it exists but hasn't been
    confirmed yet — generation should be grounded in a model the developer has
    actually reviewed, not a stale/unconfirmed draft.
    """
    if not agent_model_id:
        return None
    model = _get_store().get_agent_model(agent_model_id)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"Agent model {agent_model_id!r} not found."
        )
    if not model.confirmed:
        raise HTTPException(
            status_code=422,
            detail=f"Agent model {agent_model_id!r} has not been confirmed yet.",
        )
    return model


def _allowlist_check(number: str, settings: Settings) -> None:
    """Raise 422 if the number is not in the verified allowlist (ADR-9)."""
    try:
        check_allowlist(number, settings)
    except AllowlistError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _find_scenario(scenario_id: str):  # type: ignore[return]
    """Load scenario by id — checks builtins first, then compiled cache."""
    all_scenarios = load_all() + list(_compiled_scenarios.values())
    for s in all_scenarios:
        if s.id == scenario_id:
            return s
    raise HTTPException(status_code=404, detail=f"Scenario {scenario_id!r} not found.")


def _build_call_summary(call_row: dict, report_row: dict | None) -> CallSummary:
    return CallSummary(
        call_id=call_row["id"],
        call_index=call_row["call_index"],
        outcome=call_row.get("outcome"),
        duration_ms=call_row.get("duration_ms"),
        passed=bool(report_row["passed"]) if report_row else None,
        summary=report_row["summary"] if report_row else None,
        error=call_row.get("error"),
    )


# In-process cache of compiler-generated scenarios (keyed by scenario.id).
# Survives for the lifetime of the server process — good enough for dev/demo.
# Replace with DB-backed store when persistence is needed.
_compiled_scenarios: dict[str, Any] = {}

# In-process history of production-failure-derived regression scenarios, most
# recent last — used by include_production_regressions to grab the last 5.
_production_regression_history: list[Any] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/scenarios/compile")
async def compile_scenarios(body: CompileRequest) -> CompileResponse:
    """
    Turn a plain-English description into 3 test scenarios (easy/medium/hard).

    The generated scenarios are cached in-process so they can be referenced
    immediately in POST /runs without re-compiling.
    """
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenarios = await compiler.compile(body.description)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

    items: list[CompiledScenarioItem] = []
    dicts: list[dict[str, Any]] = []
    difficulties = ["easy", "medium", "hard"]

    for i, scenario in enumerate(scenarios):
        _compiled_scenarios[scenario.id] = scenario
        items.append(
            CompiledScenarioItem(
                id=scenario.id,
                title=scenario.title,
                category=scenario.category,
                difficulty=difficulties[i] if i < len(difficulties) else "custom",
                compiler_version=scenario.compiler_version,
            )
        )
        dicts.append(scenario.model_dump())

    return CompileResponse(scenarios=items, scenario_dicts=dicts)


@app.post("/runs", status_code=202)
async def create_run(body: CreateRunRequest) -> dict[str, str]:
    settings = _get_settings()
    _allowlist_check(body.number, settings)

    if body.n < 1:
        raise HTTPException(status_code=422, detail="n must be >= 1")

    scenario = _find_scenario(body.scenario_id)

    run_id = str(uuid.uuid4())
    store = _get_store()
    store.create_run(run_id, body.scenario_id, body.number, body.n)

    start_run(run_id, scenario, body.number, body.n, store)

    return {"run_id": run_id}


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str) -> RunStatusResponse:
    store = _get_store()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    call_rows = store.get_calls_for_run(run_id)
    summaries: list[CallSummary] = []
    for call_row in call_rows:
        report_row = store.get_report_for_call(call_row["id"])
        summaries.append(_build_call_summary(call_row, report_row))

    # Surface the first call-level error so the UI can show *why* a run failed
    # instead of a bare "failed" status (e.g. telephony not configured).
    run_error = next((s.error for s in summaries if s.error), None)

    return RunStatusResponse(
        run_id=run["id"],
        scenario_id=run["scenario_id"],
        number=run["number"],
        n=run["n"],
        status=run["status"],
        completed=run["completed"],
        calls=summaries,
        error=run_error,
    )


@app.get("/runs/{run_id}/report", response_model=RunReportResponse)
async def get_run_report(run_id: str) -> RunReportResponse:
    store = _get_store()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    report_rows = store.get_reports_for_run(run_id)

    pass_count = sum(1 for r in report_rows if r["passed"])
    completed = run["completed"]
    total = run["n"]
    pass_rate = pass_count / completed if completed > 0 else 0.0

    calls_breakdown: list[dict[str, Any]] = []
    for rpt in report_rows:
        calls_breakdown.append(
            {
                "call_id": rpt["call_id"],
                "call_index": rpt.get("call_index"),
                "outcome": rpt.get("outcome"),
                "duration_ms": rpt.get("duration_ms"),
                "passed": bool(rpt["passed"]),
                "summary": rpt["summary"],
                "dimensions": json.loads(rpt["dimensions"]),
            }
        )

    return RunReportResponse(
        run_id=run_id,
        scenario_id=run["scenario_id"],
        total=total,
        completed=completed,
        pass_count=pass_count,
        pass_rate=pass_rate,
        calls=calls_breakdown,
    )


# ---------------------------------------------------------------------------
# Agent model — "understand my agent" (Step 1: the durable artifact)
# ---------------------------------------------------------------------------


class ExtractAgentModelRequest(BaseModel):
    description: str


class ConfirmAgentModelRequest(BaseModel):
    # The (possibly developer-edited) model to persist as confirmed.
    model: dict[str, Any]


@app.post("/agent-model/extract")
async def extract_agent_model(body: ExtractAgentModelRequest) -> dict[str, Any]:
    """
    Turn a plain-English description into a DRAFT AgentModel and persist it.

    This is the keystone step: everything downstream (scenario generation,
    coverage, root-cause) will reference this object rather than the raw prompt.
    The returned model is a draft — the developer confirms/edits it next.
    """
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    try:
        extractor = _get_extractor()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        model = await extractor.extract(body.description)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Extractor error: {exc}") from exc

    _get_store().save_agent_model(model)
    return model.model_dump()


@app.get("/agent-model/{model_id}")
async def get_agent_model(model_id: str) -> dict[str, Any]:
    model = _get_store().get_agent_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Agent model {model_id!r} not found.")
    return model.model_dump()


@app.post("/agent-model/{model_id}/confirm")
async def confirm_agent_model(
    model_id: str, body: ConfirmAgentModelRequest
) -> dict[str, Any]:
    """Persist the developer-reviewed model as confirmed."""
    store = _get_store()
    if store.get_agent_model(model_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent model {model_id!r} not found.")

    try:
        model = AgentModel.model_validate(body.model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid model: {exc}") from exc

    model.id = model_id
    model.confirmed = True
    model.ensure_node_ids()
    store.save_agent_model(model)
    return model.model_dump()


# ---------------------------------------------------------------------------
# Battery — describe an agent, auto-generate + run a full stress-test suite
# ---------------------------------------------------------------------------


@app.post("/battery/preview", response_model=BatteryPreviewResponse)
async def preview_battery(body: PreviewBatteryRequest) -> BatteryPreviewResponse:
    """
    Generate scenarios from a description WITHOUT running them.
    Returns the full list so the UI can let the user pick which to run.
    Scenarios are cached in-process; pass their IDs to POST /battery.

    When `agent_model_id` names a confirmed AgentModel, generation is grounded
    in it: the prompt references its real nodes/tools/rules/risks and each
    scenario is tagged with the node_id it targets.
    """
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    agent_model = _resolve_confirmed_agent_model(body.agent_model_id)

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenarios = await compiler.compile_battery(
            body.description, body.known_issues, agent_model=agent_model
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

    for s in scenarios:
        _compiled_scenarios[s.id] = s

    return BatteryPreviewResponse(
        scenarios=[
            BatteryPreviewScenario(
                id=s.id,
                title=s.title,
                group=s.group or "other",
                category=s.category,
                node_id=s.node_id,
            )
            for s in scenarios
        ]
    )


@app.post("/battery/generate-for-node", response_model=BatteryPreviewResponse)
async def generate_for_node(body: GenerateForNodeRequest) -> BatteryPreviewResponse:
    """
    Generate additional scenarios targeting one specific node of a confirmed
    AgentModel's graph — the "generate 10 more for this node" action from the
    graph panel. Cached in-process like /battery/preview so the picker can
    pick them up immediately.
    """
    if body.count < 1:
        raise HTTPException(status_code=422, detail="count must be >= 1")

    agent_model = _resolve_confirmed_agent_model(body.agent_model_id)
    if agent_model is None:
        raise HTTPException(status_code=404, detail="agent_model_id is required.")

    if not any(n.id == body.node_id for n in agent_model.nodes):
        raise HTTPException(
            status_code=404,
            detail=f"Node {body.node_id!r} not found in agent model {body.agent_model_id!r}.",
        )

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenarios = await compiler.compile_battery(
            agent_model.source_description,
            agent_model=agent_model,
            target_node_id=body.node_id,
            count=body.count,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

    for s in scenarios:
        _compiled_scenarios[s.id] = s

    return BatteryPreviewResponse(
        scenarios=[
            BatteryPreviewScenario(
                id=s.id,
                title=s.title,
                group=s.group or "other",
                category=s.category,
                node_id=s.node_id,
            )
            for s in scenarios
        ]
    )


@app.post("/battery/generate-regression", response_model=BatteryPreviewResponse)
async def generate_regression(body: GenerateRegressionRequest) -> BatteryPreviewResponse:
    """
    Generate 5 scenarios specifically designed to trigger one root-cause
    failure mode (e.g. "Calendar API timeout") — the "Run regression test for
    this cause" action from the root-cause report section. Cached in-process
    like the other /battery/* generation endpoints.
    """
    if not body.root_cause_description.strip():
        raise HTTPException(
            status_code=422, detail="root_cause_description must not be empty"
        )

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenarios = await compiler.compile_regression(body.root_cause_description)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

    for s in scenarios:
        _compiled_scenarios[s.id] = s

    return BatteryPreviewResponse(
        scenarios=[
            BatteryPreviewScenario(
                id=s.id,
                title=s.title,
                group=s.group or "regression",
                category=s.category,
                node_id=s.node_id,
            )
            for s in scenarios
        ]
    )


@app.post("/production/ingest", response_model=IngestProductionCallsResponse)
async def ingest_production_calls(body: IngestProductionCallsRequest) -> IngestProductionCallsResponse:
    """
    Ingest production call logs: dedup on transcript, categorize new failures
    by root cause, and auto-generate 3 regression scenarios per new failure via
    the compiler (mirroring /battery/generate-regression, but driven off the
    real transcript instead of a hand-typed root-cause description).
    """
    from voicetest.scoring.root_cause import RootCauseAnalyzer

    if not body.calls:
        raise HTTPException(status_code=422, detail="calls must not be empty")

    store = _get_store()
    agent_id = body.agent_model_id or "unknown"
    analyzer = RootCauseAnalyzer()

    new_failures: list[dict[str, Any]] = []
    for call in body.calls:
        if not call.transcript.strip():
            continue
        if store.production_call_exists(agent_id, call.transcript):
            continue

        is_failure = call.outcome != "completed" or bool(call.error)
        root_cause = ""
        if is_failure:
            text = f"{call.transcript} {call.error or ''}".lower()
            root_cause, _, _ = analyzer.classify_text(text)

        call_id = str(uuid.uuid4())
        store.save_production_call(
            call_id, agent_id, call.transcript, call.outcome, call.error,
            root_cause=root_cause,
        )
        if is_failure:
            new_failures.append({"id": call_id, "transcript": call.transcript})

    regression_count = 0
    if new_failures:
        try:
            compiler = _get_compiler()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        for failure in new_failures:
            try:
                scenarios = await compiler.compile_from_failure_text(failure["transcript"])
            except Exception:  # noqa: BLE001
                continue
            for s in scenarios:
                _compiled_scenarios[s.id] = s
            _production_regression_history.extend(scenarios)
            regression_count += len(scenarios)

    return IngestProductionCallsResponse(
        new_failures=len(new_failures),
        regression_scenarios_generated=regression_count,
    )


@app.get("/production/failures", response_model=ProductionFailuresResponse)
async def get_production_failures(agent_model_id: str | None = None) -> ProductionFailuresResponse:
    """List production-derived failures, most recent first."""
    store = _get_store()
    rows = store.get_production_failures(agent_model_id)
    return ProductionFailuresResponse(
        failures=[
            ProductionFailureItem(
                id=r["id"],
                agent_id=r["agent_id"],
                transcript=r["transcript"],
                outcome=r["outcome"],
                error=r["error"],
                root_cause=r["root_cause"],
                detected_at=r["detected_at"],
            )
            for r in rows
        ]
    )


@app.post("/production/failures/generate-scenario", response_model=BatteryPreviewResponse)
async def generate_scenario_from_failure(body: GenerateScenarioFromFailureRequest) -> BatteryPreviewResponse:
    """Manually (re-)generate regression scenarios from one stored production failure."""
    store = _get_store()
    call = store.get_production_call(body.call_id)
    if call is None:
        raise HTTPException(status_code=404, detail=f"Production call {body.call_id!r} not found.")

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenarios = await compiler.compile_from_failure_text(call["transcript"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

    for s in scenarios:
        _compiled_scenarios[s.id] = s
    _production_regression_history.extend(scenarios)

    return BatteryPreviewResponse(
        scenarios=[
            BatteryPreviewScenario(
                id=s.id, title=s.title, group=s.group or "regression",
                category=s.category, node_id=s.node_id,
            )
            for s in scenarios
        ]
    )


@app.post("/battery", status_code=202)
async def create_battery(body: CreateBatteryRequest) -> dict[str, str]:
    """
    The one-step flow: describe your agent → we generate ~20 stress-test
    scenarios → run them all in parallel → aggregate into a coverage report.

    When `agent_model_id` names a confirmed AgentModel, generation is grounded
    in it and the resulting report includes per-node graph coverage.
    """
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="description must not be empty")

    settings = _get_settings()
    _allowlist_check(body.number, settings)

    agent_model = _resolve_confirmed_agent_model(body.agent_model_id)

    try:
        compiler = _get_compiler()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # If the caller pre-selected scenarios from /battery/preview, use those directly.
    if body.selected_scenario_ids:
        scenarios = [
            _compiled_scenarios[sid]
            for sid in body.selected_scenario_ids
            if sid in _compiled_scenarios
        ]
        if not scenarios:
            raise HTTPException(
                status_code=422,
                detail="None of the selected_scenario_ids were found. "
                       "Call POST /battery/preview first to generate and cache scenarios.",
            )
    else:
        try:
            scenarios = await compiler.compile_battery(
                body.description, body.known_issues, agent_model=agent_model
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Compiler error: {exc}") from exc

        if not scenarios:
            raise HTTPException(status_code=500, detail="Compiler produced no scenarios.")

        # Cache the generated scenarios so per-scenario lookups still resolve.
        for s in scenarios:
            _compiled_scenarios[s.id] = s

    if body.include_production_regressions and _production_regression_history:
        scenarios = list(scenarios) + _production_regression_history[-5:]

    battery_id = str(uuid.uuid4())
    store = _get_store()
    store.create_battery(
        battery_id, body.description, body.number, len(scenarios),
        agent_model_id=agent_model.id if agent_model else None,
    )
    start_battery(battery_id, scenarios, body.number, store, agent_model=agent_model)

    return {"battery_id": battery_id, "total": str(len(scenarios))}


def _empty_group() -> dict[str, int]:
    return {"total": 0, "passed": 0, "failed": 0, "running": 0}


@app.get("/battery/{battery_id}", response_model=BatteryStatusResponse)
async def get_battery_status(battery_id: str) -> BatteryStatusResponse:
    store = _get_store()
    battery = store.get_battery(battery_id)
    if battery is None:
        raise HTTPException(status_code=404, detail=f"Battery {battery_id!r} not found.")

    runs = store.get_runs_for_battery(battery_id)
    groups: dict[str, dict[str, int]] = {}
    for run in runs:
        grp = run.get("scenario_group") or "other"
        g = groups.setdefault(grp, _empty_group())
        g["total"] += 1

        report_rows = store.get_reports_for_run(run["id"])
        if run["status"] == RunStatus.COMPLETED and report_rows:
            if all(r["passed"] for r in report_rows):
                g["passed"] += 1
            else:
                g["failed"] += 1
        elif run["status"] == RunStatus.FAILED:
            g["failed"] += 1
        else:
            g["running"] += 1

    return BatteryStatusResponse(
        battery_id=battery_id,
        description=battery["description"],
        status=battery["status"],
        total=battery["total"],
        completed=battery["completed"],
        groups={k: GroupProgress(**v) for k, v in groups.items()},
    )


@app.get("/battery/{battery_id}/report", response_model=BatteryReportResponse)
async def get_battery_report(battery_id: str) -> BatteryReportResponse:
    store = _get_store()
    battery = store.get_battery(battery_id)
    if battery is None:
        raise HTTPException(status_code=404, detail=f"Battery {battery_id!r} not found.")

    report = (
        json.loads(battery["report_json"]) if battery.get("report_json") else None
    )
    return BatteryReportResponse(
        battery_id=battery_id,
        status=battery["status"],
        report=report,
    )


@app.get("/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    builtin = get_builtin_scenarios()
    compiled = list(_compiled_scenarios.values())
    all_scenarios = builtin + compiled
    return [
        {
            "id": s.id,
            "title": s.title,
            "category": s.category,
            "version": s.version,
            "compiler_version": s.compiler_version,
        }
        for s in all_scenarios
    ]


@app.get("/runs/{run_id}/transcript")
async def get_run_transcript(run_id: str) -> list[dict[str, Any]]:
    """Per-call transcripts (turns list) for rendering in the UI."""
    store = _get_store()
    if store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")

    call_rows = store.get_calls_for_run(run_id)
    result: list[dict[str, Any]] = []
    for row in call_rows:
        turns: list[dict[str, Any]] = []
        if row.get("result_json"):
            data = json.loads(row["result_json"])
            turns = data.get("turns", [])
        result.append({"call_id": row["id"], "call_index": row["call_index"], "turns": turns})
    return result


@app.websocket("/twilio-stream/{stream_id}")
async def twilio_stream_endpoint(websocket: WebSocket, stream_id: str) -> None:
    """Twilio Media Streams connects here; bridge audio to the engine pipeline."""
    from voicetest.engine.ws_server import twilio_stream
    from voicetest.log import vlog

    vlog("ws", f"/twilio-stream/{stream_id} endpoint HIT (incoming WS handshake)")
    await twilio_stream(websocket, stream_id)


@app.get("/recordings/{call_id}")
async def get_recording(call_id: str) -> FileResponse:
    store = _get_store()
    call = store.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail=f"Call {call_id!r} not found.")

    result_json = call.get("result_json")
    recording_path: str | None = None
    if result_json:
        data = json.loads(result_json)
        recording_path = data.get("recording_path")

    if not recording_path or not Path(recording_path).exists():
        raise HTTPException(status_code=404, detail="Recording not available for this call.")

    return FileResponse(recording_path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# UI — single-page HTML app (no build step, no external CDN)
# ---------------------------------------------------------------------------

_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>voicetest</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --grad-start: #6366f1;
    --grad-end: #8b5cf6;
    --grad: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    --grad-soft: linear-gradient(135deg, #eef2ff, #f5f3ff);
    --bg: #f8f7ff;
    --surface: #ffffff;
    --surface-2: #f4f3ff;
    --border: #e5e4f0;
    --text: #1e1b4b;
    --text-2: #6b6a8e;
    --text-3: #9ca3af;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --pass-bg: #dcfce7; --pass-fg: #166534;
    --fail-bg: #fee2e2; --fail-fg: #991b1b;
    --pending-bg: #f1f5f9; --pending-fg: #475569;
    --easy-bg: #cffafe; --easy-fg: #155e75;
    --medium-bg: #fef9c3; --medium-fg: #854d0e;
    --hard-bg: #ffe4e6; --hard-fg: #9f1239;
    --radius: 14px;
    --radius-sm: 8px;
    --shadow: 0 1px 3px rgba(99,102,241,.08), 0 4px 16px rgba(99,102,241,.06);
    --shadow-md: 0 4px 24px rgba(99,102,241,.12);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Montserrat', system-ui, sans-serif;
    font-size: 14px;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Layout ── */
  .app-header {
    background: var(--grad);
    padding: 0 32px;
    display: flex;
    align-items: center;
    gap: 12px;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 20px rgba(99,102,241,.25);
  }
  .app-header .logo-icon {
    width: 32px; height: 32px;
    background: rgba(255,255,255,.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .app-header h1 {
    color: #fff;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.3px;
  }
  .app-header .tagline {
    color: rgba(255,255,255,.65);
    font-size: 12px;
    font-weight: 500;
    margin-left: 4px;
  }

  .page-body {
    max-width: 900px;
    margin: 0 auto;
    padding: 32px 20px 60px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
  }
  .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
  }
  .card-icon {
    width: 36px; height: 36px;
    border-radius: var(--radius-sm);
    background: var(--grad-soft);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
  }
  .card-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.2px;
  }
  .card-subtitle {
    font-size: 12px;
    color: var(--text-2);
    font-weight: 400;
    margin-top: 2px;
  }

  h2 {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 20px 0 10px;
  }
  h2:first-child { margin-top: 0; }

  /* ── Form elements ── */
  label {
    display: block;
    margin-bottom: 6px;
    font-weight: 600;
    font-size: 12px;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  input, select, textarea {
    width: 100%;
    padding: 10px 14px;
    border: 1.5px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-family: inherit;
    font-weight: 500;
    color: var(--text);
    background: var(--surface);
    transition: border-color .15s, box-shadow .15s;
    outline: none;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(99,102,241,.12);
  }
  input::placeholder, textarea::placeholder { color: var(--text-3); font-weight: 400; }
  textarea { resize: vertical; min-height: 88px; line-height: 1.6; }
  select { appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b6a8e' d='M6 8L1 3h10z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 36px; }

  /* ── Buttons ── */
  button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    background: var(--grad);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-family: inherit;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.2px;
    transition: opacity .15s, transform .1s, box-shadow .15s;
    box-shadow: 0 2px 8px rgba(99,102,241,.3);
  }
  button:hover:not(:disabled) {
    opacity: .92;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,102,241,.4);
  }
  button:active:not(:disabled) { transform: translateY(0); }
  button:disabled { background: #d1d5db; color: #9ca3af; box-shadow: none; cursor: default; }
  button.secondary {
    background: var(--surface);
    color: var(--accent);
    border: 1.5px solid var(--border);
    box-shadow: none;
  }
  button.secondary:hover:not(:disabled) {
    background: var(--grad-soft);
    border-color: var(--accent);
    box-shadow: none;
  }

  /* ── Form grid ── */
  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr 100px;
    gap: 14px;
    align-items: end;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) {
    .form-row { grid-template-columns: 1fr; }
  }

  /* ── Progress ── */
  .progress-wrap {
    background: var(--surface-2);
    border-radius: 99px;
    height: 8px;
    margin: 12px 0 16px;
    overflow: hidden;
  }
  .progress-bar {
    background: var(--grad);
    height: 100%;
    border-radius: 99px;
    transition: width .4s cubic-bezier(.4,0,.2,1);
  }
  .progress-label {
    font-size: 13px;
    color: var(--text-2);
    font-weight: 500;
    margin-bottom: 4px;
  }
  .progress-label strong { color: var(--text); }

  /* ── Call rows ── */
  .calls-grid { display: flex; flex-direction: column; gap: 6px; }
  .call-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 500;
  }
  .call-row .call-dur { color: var(--text-3); font-size: 12px; margin-left: auto; }
  .call-row .call-err { color: #dc2626; font-size: 12px; }

  /* ── Badges ── */
  .badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .pass    { background: var(--pass-bg);    color: var(--pass-fg); }
  .fail    { background: var(--fail-bg);    color: var(--fail-fg); }
  .pending { background: var(--pending-bg); color: var(--pending-fg); }
  .easy    { background: var(--easy-bg);    color: var(--easy-fg); }
  .medium  { background: var(--medium-bg);  color: var(--medium-fg); }
  .hard    { background: var(--hard-bg);    color: var(--hard-fg); }
  .compiled-tag { background: #ede9fe; color: #5b21b6; }

  /* ── Report summary ── */
  .report-summary-row {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 20px;
    background: var(--grad-soft);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .pass-rate-big {
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -2px;
  }
  .pass-rate-big.pass { color: #16a34a; }
  .pass-rate-big.fail { color: #dc2626; }
  .pass-rate-label { font-size: 13px; color: var(--text-2); font-weight: 500; margin-top: 4px; }

  /* ── Dim grid ── */
  .dim-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 10px;
    margin-top: 8px;
  }
  .dim-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    font-size: 12px;
  }
  .dim-name {
    font-weight: 700;
    margin-bottom: 4px;
    text-transform: capitalize;
    color: var(--text);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .dim-score { font-size: 18px; margin-bottom: 4px; }
  .dim-reason { color: var(--text-2); font-size: 12px; line-height: 1.5; }

  /* ── Transcript ── */
  .transcript {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .turn {
    padding: 8px 12px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    line-height: 1.55;
  }
  .turn.caller {
    background: #eff6ff;
    color: #1d4ed8;
    align-self: flex-start;
    max-width: 85%;
  }
  .turn.agent {
    background: var(--surface-2);
    color: var(--text);
    align-self: flex-end;
    max-width: 85%;
  }
  .turn .speaker-tag {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: .6;
    display: block;
    margin-bottom: 3px;
  }
  .turn.highlight {
    background: #fef9c3;
    border-left: 3px solid #eab308;
    padding-left: 10px;
  }

  /* ── Fix boxes ── */
  .fix-box {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    font-size: 13px;
    margin-top: 8px;
  }
  .fix-label {
    font-weight: 700;
    color: #c2410c;
    margin-bottom: 4px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .fix-text { color: var(--text-2); line-height: 1.5; }

  /* ── Scenario chips ── */
  .compiled-scenarios { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
  .compiled-scenario-chip {
    border: 1.5px solid #c4b5fd;
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    background: #faf5ff;
    cursor: pointer;
    font-size: 13px;
    transition: background .15s, border-color .15s, box-shadow .15s;
    min-width: 160px;
  }
  .compiled-scenario-chip:hover {
    background: #ede9fe;
    border-color: #7c3aed;
    box-shadow: 0 2px 8px rgba(124,58,237,.15);
  }
  .compiled-scenario-chip.selected {
    border-color: #7c3aed;
    background: #ede9fe;
    box-shadow: 0 0 0 3px rgba(124,58,237,.15);
  }
  .chip-title { font-weight: 700; color: var(--text); margin-bottom: 5px; }
  .chip-meta { color: var(--text-2); font-size: 11px; display: flex; align-items: center; gap: 5px; }

  /* ── Details / accordion ── */
  details.call-detail { margin-bottom: 10px; }
  details.call-detail > summary {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    list-style: none;
    transition: background .15s;
    user-select: none;
  }
  details.call-detail > summary::-webkit-details-marker { display: none; }
  details.call-detail > summary:hover { background: var(--surface-2); }
  label.scenario-label:hover { background: var(--surface-2); }
  details.call-detail[open] > summary {
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    background: var(--surface-2);
    border-bottom-color: transparent;
  }
  .summary-chevron {
    margin-left: auto;
    color: var(--text-3);
    font-size: 12px;
    transition: transform .2s;
  }
  details.call-detail[open] .summary-chevron { transform: rotate(180deg); }
  .call-detail-body {
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm);
    padding: 18px;
    background: var(--surface);
  }

  /* ── Audio ── */
  audio { width: 100%; margin-top: 8px; border-radius: var(--radius-sm); }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    width: 15px; height: 15px;
    border: 2px solid rgba(99,102,241,.25);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .65s linear infinite;
    vertical-align: middle;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Generating progress ── */
  .gen-progress {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--grad-soft);
    padding: 16px 18px;
    margin-top: 16px;
  }
  .gen-progress-head {
    display: flex; align-items: center; gap: 10px;
    font-size: 13px; font-weight: 700; color: var(--text);
  }
  .gen-progress-sub {
    font-size: 12px; color: var(--text-2); font-weight: 500;
    margin-top: 4px; margin-left: 25px;
    min-height: 16px;
  }
  .gen-progress-timer { margin-left: auto; font-size: 12px; font-weight: 600; color: var(--text-3); font-variant-numeric: tabular-nums; }
  .gen-bar-track { height: 5px; border-radius: 99px; background: rgba(99,102,241,.15); overflow: hidden; margin-top: 12px; }
  .gen-bar-fill {
    height: 100%; width: 40%; border-radius: 99px; background: var(--grad);
    animation: gen-slide 1.4s ease-in-out infinite;
  }
  @keyframes gen-slide {
    0%   { margin-left: -40%; }
    100% { margin-left: 100%; }
  }
  .skeleton-row {
    height: 44px; border-radius: var(--radius-sm);
    background: linear-gradient(90deg, var(--surface-2) 25%, #eef0f6 50%, var(--surface-2) 75%);
    background-size: 200% 100%;
    animation: skel 1.3s linear infinite;
    margin-bottom: 6px;
  }
  @keyframes skel { to { background-position: -200% 0; } }
  .icon-inline { display: inline-flex; vertical-align: -2px; }

  /* ── Error / status msgs ── */
  .error-msg { color: #dc2626; font-size: 13px; font-weight: 500; margin-top: 10px; }
  .run-id-pill {
    display: inline-block;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 11px;
    color: var(--text-2);
    font-weight: 600;
    font-family: monospace;
    margin-left: 8px;
  }

  /* ── Section dividers ── */
  .section-sep { height: 1px; background: var(--border); margin: 18px 0; }

  #status-area { display: none; }
  #error-msg { color: #dc2626; font-size: 13px; font-weight: 500; margin-top: 10px; }

  /* ── Group matrix (live) ── */
  .group-matrix { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
  .group-row {
    display: grid;
    grid-template-columns: 1fr auto auto;
    align-items: center;
    gap: 12px;
    padding: 11px 14px;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 600;
  }
  .group-name { text-transform: capitalize; color: var(--text); }
  .group-counts { display: flex; gap: 6px; align-items: center; }
  .group-progress { color: var(--text-3); font-size: 12px; font-variant-numeric: tabular-nums; }
  .pill-pass { color: var(--pass-fg); background: var(--pass-bg); padding: 1px 8px; border-radius: 99px; font-size: 11px; font-weight: 700; }
  .pill-fail { color: var(--fail-fg); background: var(--fail-bg); padding: 1px 8px; border-radius: 99px; font-size: 11px; font-weight: 700; }
  .pill-run  { color: var(--pending-fg); background: var(--pending-bg); padding: 1px 8px; border-radius: 99px; font-size: 11px; font-weight: 700; }

  /* ── Verdict banner ── */
  .verdict-banner {
    display: flex; align-items: center; gap: 20px;
    padding: 22px 24px; border-radius: var(--radius-sm);
    margin-bottom: 16px; border: 1px solid var(--border);
  }
  .verdict-banner.ready    { background: #f0fdf4; border-color: #bbf7d0; }
  .verdict-banner.needs    { background: #fffbeb; border-color: #fde68a; }
  .verdict-banner.notready { background: #fef2f2; border-color: #fecaca; }
  .verdict-pct { font-size: 46px; font-weight: 800; line-height: 1; letter-spacing: -2px; }
  .verdict-banner.ready .verdict-pct    { color: #16a34a; }
  .verdict-banner.needs .verdict-pct    { color: #d97706; }
  .verdict-banner.notready .verdict-pct { color: #dc2626; }
  .verdict-label { font-size: 18px; font-weight: 800; color: var(--text); }
  .verdict-sub { font-size: 13px; color: var(--text-2); font-weight: 500; margin-top: 3px; }

  /* ── Critical failures ── */
  .critical-fail {
    background: #fff7ed; border: 1px solid #fed7aa;
    border-left: 4px solid #ea580c;
    border-radius: var(--radius-sm); padding: 14px 16px; margin-bottom: 10px;
  }
  .critical-fail-title { font-weight: 700; color: #c2410c; font-size: 14px; margin-bottom: 4px; }
  .critical-fail-meta { font-size: 12px; color: var(--text-2); font-weight: 500; margin-bottom: 8px; }
  .critical-fail-ev { font-size: 12px; color: var(--text-2); line-height: 1.5; padding: 4px 0 4px 14px; border-left: 2px solid #fed7aa; margin-top: 4px; }

  /* ── Root cause grouping ── */
  details.root-cause-group { margin-bottom: 12px; }
  details.root-cause-group > summary {
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; list-style: none; padding: 10px 14px;
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); font-weight: 600;
  }
  details.root-cause-group > summary::-webkit-details-marker { display: none; }
  .root-cause-title { font-size: 13px; color: var(--text); }
  details.root-cause-group[open] .summary-chevron { transform: rotate(180deg); }
  .root-cause-body { padding: 12px 4px 4px; }
  .root-cause-fix {
    font-size: 12px; color: var(--text-2); background: #eff6ff;
    border: 1px solid #bfdbfe; border-radius: var(--radius-sm);
    padding: 8px 12px; margin-bottom: 10px;
  }
  .root-cause-regression-result { margin-top: 8px; font-size: 12px; color: var(--text-2); }
  .root-cause-regression-list { margin-top: 4px; }

  /* ── Coverage chips ── */
  .coverage-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 10px; margin-top: 8px; }
  .coverage-cell {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 14px;
  }
  .coverage-dim { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .4px; color: var(--text-2); margin-bottom: 6px; }
  .coverage-stat { font-size: 13px; font-weight: 600; color: var(--text); }
  .coverage-cell.crit { border-color: #fca5a5; background: #fef2f2; }

  /* ── Agent model ── */
  .am-goal {
    background: var(--grad-soft); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 16px; margin-bottom: 14px;
    font-size: 14px; font-weight: 600; color: var(--text);
  }
  .am-section { margin-bottom: 14px; }
  .am-section > .am-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .5px;
    color: var(--text-3); margin-bottom: 6px;
  }
  .am-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .am-chip {
    background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 99px; padding: 4px 12px; font-size: 12px; font-weight: 600; color: var(--text);
  }
  .am-chip.risk { background: #fff7ed; border-color: #fed7aa; color: #c2410c; }
  .am-chip.rule { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
  .am-list { display: flex; flex-direction: column; gap: 5px; }
  .am-list-item { font-size: 13px; color: var(--text-2); line-height: 1.5; padding-left: 16px; position: relative; }
  .am-list-item::before { content: '✓'; position: absolute; left: 0; color: var(--accent); font-weight: 700; }
  /* graph as a simple flow */
  .am-graph { display: flex; flex-direction: column; gap: 4px; }
  .am-graph-node {
    background: var(--surface); border: 1.5px solid #c4b5fd;
    border-radius: var(--radius-sm); padding: 8px 14px; font-size: 13px; font-weight: 700;
    color: var(--text); display: inline-flex; align-items: center; gap: 8px; align-self: flex-start;
  }
  .am-graph-edge { font-size: 11px; color: var(--text-3); padding-left: 14px; font-weight: 600; }
  .am-graph-edge .cond { color: var(--accent); }

  /* ── Section nav ── */
  #section-nav {
    max-width: 900px;
    margin: 24px auto 0;
    padding: 0 20px;
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border);
  }
  .nav-tab {
    background: transparent;
    color: var(--text-2);
    border: none;
    box-shadow: none;
    border-radius: 0;
    padding: 12px 16px;
    font-size: 13px;
    font-weight: 700;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }
  .nav-tab:hover:not(:disabled) {
    background: transparent;
    color: var(--text);
    transform: none;
    box-shadow: none;
  }
  .nav-tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  .section-hidden { display: none !important; }

  /* ── Stepper ── */
  .stepper {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 22px;
  }
  .step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-3);
    white-space: nowrap;
  }
  .step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: var(--surface-2);
    border: 1.5px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
    color: var(--text-3);
    flex-shrink: 0;
    transition: background .2s, border-color .2s, color .2s;
  }
  .step.active { color: var(--accent); }
  .step.active .step-num { background: var(--accent); border-color: var(--accent); color: #fff; }
  .step.done { color: #16a34a; }
  .step.done .step-num { background: #16a34a; border-color: #16a34a; color: #fff; }
  .step-line {
    flex: 1;
    height: 2px;
    background: var(--border);
    margin: 0 10px;
    min-width: 20px;
  }
  .step.done + .step-line { background: #16a34a; }

  /* ── Node graph (SVG, layered) ── */
  .graph-panel {
    overflow: auto;
    padding: 8px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background:
      radial-gradient(circle at 1px 1px, rgba(99,102,241,.09) 1px, transparent 0);
    background-size: 22px 22px;
    background-color: var(--surface-2);
  }
  .graph-svg { display: block; }
  .graph-node-fo { cursor: pointer; }
  .graph-node-box {
    background: var(--surface);
    border: 2px solid #c4b5fd;
    border-radius: 12px;
    padding: 10px 12px;
    box-shadow: var(--shadow);
    transition: border-color .15s, box-shadow .15s;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
  }
  .graph-node-fo:hover .graph-node-box {
    border-color: #7c3aed;
    box-shadow: 0 4px 16px rgba(124,58,237,.25);
  }
  .graph-node-box.selected {
    border-color: #7c3aed;
    background: #ede9fe;
    box-shadow: 0 0 0 3px rgba(124,58,237,.18);
  }
  .graph-node-box .gn-label {
    font-size: 12.5px; font-weight: 800; color: var(--text); line-height: 1.25;
  }
  .graph-node-box .gn-cov {
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 6px; font-size: 10px; font-weight: 700;
  }
  .gn-cov .gn-dot { width: 7px; height: 7px; border-radius: 50%; }
  .gn-cov.covered { color: #16a34a; } .gn-cov.covered .gn-dot { background: #16a34a; }
  .gn-cov.empty   { color: var(--text-3); } .gn-cov.empty .gn-dot { background: #cbd5e1; }
  .graph-edge-path { fill: none; stroke: #b4b0d8; stroke-width: 1.75; }
  .graph-edge-label {
    font-size: 9.5px; font-weight: 700; fill: var(--accent);
    text-transform: uppercase; letter-spacing: .3px;
  }
  .graph-edge-label-bg { fill: var(--surface-2); }

  /* ── Node detail side panel ── */
  .node-panel-overlay {
    position: fixed; inset: 0;
    background: rgba(30,27,75,.35);
    z-index: 200;
    display: flex; justify-content: flex-end;
  }
  .node-panel {
    width: 360px;
    max-width: 90vw;
    height: 100%;
    background: var(--surface);
    box-shadow: -8px 0 32px rgba(30,27,75,.2);
    padding: 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .node-panel-close {
    align-self: flex-end;
    background: var(--surface-2);
    color: var(--text-2);
    border: none;
    box-shadow: none;
    padding: 6px 10px;
    font-size: 12px;
  }
  .node-panel-title { font-size: 17px; font-weight: 800; color: var(--text); }
  .node-panel-objective { font-size: 13px; color: var(--text-2); line-height: 1.5; }
  .node-panel-stat-row { display: flex; gap: 10px; }
  .node-panel-stat {
    flex: 1;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    text-align: center;
  }
  .node-panel-stat .stat-num { font-size: 20px; font-weight: 800; color: var(--text); }
  .node-panel-stat .stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: .4px; color: var(--text-3); margin-top: 2px; }
  .node-panel-stat.pass .stat-num { color: #16a34a; }
  .node-panel-stat.fail .stat-num { color: #dc2626; }
  .node-scenario-list { display: flex; flex-direction: column; gap: 6px; }
  .node-scenario-row {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 600;
  }
  #generate-for-node-error { color: #dc2626; font-size: 12px; font-weight: 500; }
</style>
</head>
<body>

<header class="app-header">
  <div class="logo-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg></div>
  <h1>voicetest</h1>
  <span class="tagline">Voice AI testing platform</span>
</header>

<!-- ── Section nav (tabs) ─────────────────────────────────────────────────── -->
<nav id="section-nav">
  <button class="nav-tab active" data-section="build" onclick="switchSection('build')">Build &amp; test</button>
  <button class="nav-tab" data-section="load" onclick="switchSection('load')">Load report</button>
  <button class="nav-tab" data-section="production" onclick="switchSection('production')">Production learning</button>
</nav>

<div class="page-body">

<div id="section-build">
<!-- ── Stress-test battery (guided funnel: describe → confirm → scenarios → run) ── -->
<div class="card">
  <div class="card-header">
    <div class="card-icon" data-icon="sparkles"></div>
    <div>
      <div class="card-title">Stress-test your voice agent</div>
      <div class="card-subtitle">Describe what your agent does. We understand it, generate realistic test scenarios, and run them.</div>
    </div>
  </div>

  <!-- Guided stepper -->
  <div class="stepper" id="stepper">
    <div class="step active" data-step="1"><span class="step-num">1</span><span>Describe</span></div>
    <div class="step-line"></div>
    <div class="step" data-step="2"><span class="step-num">2</span><span>Confirm model</span></div>
    <div class="step-line"></div>
    <div class="step" data-step="3"><span class="step-num">3</span><span>Pick scenarios</span></div>
    <div class="step-line"></div>
    <div class="step" data-step="4"><span class="step-num">4</span><span>Run</span></div>
  </div>

  <!-- Step 1: describe -->
  <div id="step1">
    <label for="battery-desc">What does your agent do?</label>
    <textarea id="battery-desc" rows="4"
      placeholder="e.g. My agent answers calls for a hair salon. It books appointments — collecting the customer's name, the date and time they want, and a callback number — and can reschedule existing bookings."></textarea>

    <div style="margin-top:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <button id="understand-btn" onclick="understandAgent()"><span data-icon="brain"></span> Understand my agent</button>
      <button id="preview-btn" class="secondary" onclick="previewScenarios()"><span data-icon="search"></span> Skip to scenarios</button>
    </div>
    <div id="preview-error" class="error-msg"></div>
    <div id="understand-progress"></div>
  </div>

  <!-- Agent model: "We understand your agent" (hidden until extracted) -->
  <div id="agent-model" style="display:none;margin-top:20px">
    <div class="section-sep"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
      <div style="font-size:14px;font-weight:800;color:var(--text);display:flex;align-items:center;gap:8px"><span data-icon="brain"></span> We understand your agent</div>
      <span id="am-confirmed-badge" class="badge pending">Draft</span>
    </div>
    <div class="card-subtitle" style="margin-bottom:14px">A draft model extracted from your description. Review it — everything we test is grounded in this.</div>
    <div id="am-body"></div>
    <div style="margin-top:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <button id="am-confirm-btn" onclick="confirmAgentModel()"><span data-icon="check"></span> Looks right — confirm &amp; generate scenarios</button>
      <span id="am-spinner" style="display:none;align-items:center;gap:6px;color:var(--text-2);font-size:13px;font-weight:500">
        <span class="spinner"></span> Saving…
      </span>
    </div>
    <div id="am-error" class="error-msg"></div>
  </div>

  <!-- Node graph panel (visual graph, shown once confirmed) -->
  <div id="node-graph-section" style="display:none;margin-top:20px">
    <div class="section-sep"></div>
    <div style="font-size:14px;font-weight:800;color:var(--text);margin-bottom:6px;display:flex;align-items:center;gap:8px"><span data-icon="graph"></span> Conversation graph</div>
    <div class="card-subtitle" style="margin-bottom:10px">Click a node to see its test coverage and generate more scenarios for it. Green nodes have scenarios.</div>
    <div class="graph-panel"><div id="graph-panel-body"></div></div>
  </div>
  <div id="node-panel-container"></div>

  <!-- Scenario generation progress (shown while generating) -->
  <div id="scenario-progress" style="display:none"></div>

  <!-- Step 2: scenario picker (hidden until generated) -->
  <div id="step2" style="display:none;margin-top:20px">
    <div class="section-sep"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <div style="font-size:14px;font-weight:800;color:var(--text)">Choose scenarios to run <span id="scenario-count" style="font-weight:600;color:var(--text-3);font-size:13px"></span></div>
      <div style="display:flex;gap:8px">
        <button class="secondary" style="padding:6px 12px;font-size:12px" onclick="selectAll(true)">Select all</button>
        <button class="secondary" style="padding:6px 12px;font-size:12px" onclick="selectAll(false)">Deselect all</button>
      </div>
    </div>
    <div id="scenario-picker"></div>

    <div style="margin-top:16px;display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap">
      <div style="flex:1;min-width:180px">
        <label for="battery-number">Phone number <span style="text-transform:none;font-weight:400;color:var(--text-3)">(loopback = no telephony)</span></label>
        <input id="battery-number" type="text" placeholder="loopback or +15551234567" value="loopback">
      </div>
      <button id="battery-btn" onclick="startBattery()"><span data-icon="bolt"></span> Run selected scenarios</button>
      <span id="battery-spinner" style="display:none;align-items:center;gap:6px;color:var(--text-2);font-size:13px;font-weight:500">
        <span class="spinner"></span> <span id="battery-spinner-text">Starting…</span>
      </span>
    </div>
    <div id="battery-error" class="error-msg"></div>
  </div>
</div>
</div><!-- /section-build -->

<div id="section-load" class="section-hidden">
<!-- ── Load existing report ────────────────────────────────────────────── -->
<div class="card">
  <div class="card-header">
    <div class="card-icon" data-icon="clipboard"></div>
    <div>
      <div class="card-title">Load existing report</div>
      <div class="card-subtitle">Retrieve a completed battery by its ID</div>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap">
    <div style="flex:1;min-width:200px">
      <label for="load-battery-id">Battery ID</label>
      <input id="load-battery-id" type="text" placeholder="paste a battery UUID here">
    </div>
    <button class="secondary" onclick="loadBatteryById()">Load report</button>
  </div>
  <div id="load-error-msg" class="error-msg"></div>
</div>
</div><!-- /section-load -->

<div id="section-production" class="section-hidden">
<!-- ── Production Learning ──────────────────────────────────────────────── -->
<div class="card">
  <div class="card-header">
    <div class="card-icon" data-icon="phone"></div>
    <div>
      <div class="card-title">Production learning</div>
      <div class="card-subtitle">Paste production call logs — we dedup, categorize new failures, and auto-generate regression tests.</div>
    </div>
  </div>
  <label for="production-calls-input">Call logs</label>
  <textarea id="production-calls-input" rows="6"
    placeholder='JSON array: [{"transcript": "...", "outcome": "error", "error": "timeout"}] or plain text transcripts, one per line'></textarea>
  <div style="margin-top:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
    <button id="ingest-btn" onclick="ingestProductionCalls()"><span data-icon="download"></span> Ingest production calls</button>
    <span id="ingest-spinner" style="display:none;align-items:center;gap:6px;color:var(--text-2);font-size:13px;font-weight:500">
      <span class="spinner"></span> Working…
    </span>
    <a href="#" onclick="loadProductionFailures();return false;" style="font-size:13px;font-weight:600">View all production failures →</a>
  </div>
  <div id="ingest-result" style="margin-top:10px;font-size:13px;font-weight:600;color:var(--text)"></div>
  <div id="ingest-error" class="error-msg"></div>

  <div id="production-failures-list" style="display:none;margin-top:18px">
    <div class="section-sep"></div>
    <div style="font-size:14px;font-weight:800;color:var(--text);margin-bottom:10px">Production failures</div>
    <div id="production-failures-body"></div>
  </div>
</div>
</div><!-- /section-production -->

<!-- ── Live status + report ────────────────────────────────────────────── -->
<div id="status-area">
  <div class="card">
    <div class="card-header">
      <div class="card-icon" data-icon="radio"></div>
      <div>
        <div class="card-title">Stress test running<span id="run-id-label" class="run-id-pill"></span></div>
        <div class="card-subtitle" id="progress-text">Starting…</div>
      </div>
    </div>
    <div class="progress-wrap">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
    <div id="group-matrix"></div>
  </div>

  <div class="card" id="report-card" style="display:none">
    <div class="card-header">
      <div class="card-icon" data-icon="chart"></div>
      <div>
        <div class="card-title">Stress-test report</div>
        <div class="card-subtitle">What your agent handled — and what broke</div>
      </div>
    </div>
    <div id="report-verdict"></div>
    <div id="report-critical"></div>
    <div id="report-graph-coverage"></div>
    <div id="report-coverage"></div>
    <div id="report-scenarios"></div>
  </div>
</div>

</div><!-- /page-body -->

<script>
// ── icon set (inline SVG, no CDN) ───────────────────────────────────────────
const ICONS = {
  sparkles: '<path d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"/><path d="M19 14l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/>',
  brain: '<path d="M9.5 4a2.5 2.5 0 0 0-2.5 2.5c-1.2.3-2 1.4-2 2.7 0 .5.1 1 .4 1.4A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 9.5 17V4z"/><path d="M14.5 4A2.5 2.5 0 0 1 17 6.5c1.2.3 2 1.4 2 2.7 0 .5-.1 1-.4 1.4A2.5 2.5 0 0 1 19 15.5 2.5 2.5 0 0 1 14.5 17V4z"/>',
  search: '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  graph: '<rect x="3" y="4" width="7" height="5" rx="1.5"/><rect x="14" y="4" width="7" height="5" rx="1.5"/><rect x="8.5" y="15" width="7" height="5" rx="1.5"/><path d="M6.5 9v3.5h5.5V15"/><path d="M17.5 9v3.5h-5.5"/>',
  bolt: '<polygon points="13 2 3 14 11 14 11 22 21 10 13 10 13 2"/>',
  clipboard: '<rect x="8" y="3" width="8" height="4" rx="1"/><path d="M8 5H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>',
  phone: '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3 19.5 19.5 0 0 1-6-6 19.8 19.8 0 0 1-3-8.6A2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.3-1.2a2 2 0 0 1 2.1-.5c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
  radio: '<circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.5M7.8 16.2a6 6 0 0 1 0-8.5M19 5a9 9 0 0 1 0 14M5 19A9 9 0 0 1 5 5"/>',
  chart: '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
};
function iconSvg(name, size) {
  const s = size || 15;
  const p = ICONS[name] || '';
  return '<svg class="icon-inline" width="'+s+'" height="'+s+'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+p+'</svg>';
}
function hydrateIcons() {
  document.querySelectorAll('[data-icon]').forEach(el => {
    if (el.dataset.iconDone) return;
    const isCard = el.classList.contains('card-icon');
    el.innerHTML = iconSvg(el.getAttribute('data-icon'), isCard ? 17 : 15);
    if (isCard) el.style.color = 'var(--accent)';
    el.dataset.iconDone = '1';
  });
}

// ── section nav ─────────────────────────────────────────────────────────────
function switchSection(name) {
  ['build','load','production'].forEach(s => {
    document.getElementById('section-' + s).classList.toggle('section-hidden', s !== name);
  });
  document.querySelectorAll('.nav-tab').forEach(t =>
    t.classList.toggle('active', t.getAttribute('data-section') === name));
}

// ── stepper ─────────────────────────────────────────────────────────────────
// mark steps < n as done, step n as active
function setStep(n) {
  document.querySelectorAll('#stepper .step').forEach(el => {
    const s = +el.getAttribute('data-step');
    el.classList.toggle('done', s < n);
    el.classList.toggle('active', s === n);
  });
}

document.addEventListener('DOMContentLoaded', hydrateIcons);

// ── battery lifecycle ──────────────────────────────────────────────────────
let _batteryTimer = null;
let _batteryReportLoaded = false;
let _previewScenarios = [];  // all scenarios from last preview call

const GROUP_ORDER = ['baseline','confused_caller','impatient_caller','edge_cases','audio_stress','adversarial','regression','other'];
const DIM_LABELS = {goal_completion:'Goal completion', barge_in_handling:'Barge-in', noise_resilience:'Noise resilience', latency:'Latency'};

function showStatusArea() {
  const sa = document.getElementById('status-area');
  sa.style.display = 'flex';
  sa.style.flexDirection = 'column';
  sa.style.gap = '20px';
}

function selectAll(checked) {
  document.querySelectorAll('#scenario-picker input[type=checkbox]').forEach(cb => cb.checked = checked);
}

// ── live progress widget ────────────────────────────────────────────────────
// Renders an animated progress card into `containerId`, cycles through `stages`
// on a timer, and shows an elapsed clock. Returns a stop() to call on completion.
function startProgress(containerId, title, stages, opts) {
  opts = opts || {};
  const el = document.getElementById(containerId);
  el.style.display = '';
  el.innerHTML =
    '<div class="gen-progress">' +
      '<div class="gen-progress-head"><span class="spinner"></span><span>' + escHtml(title) + '</span>' +
        '<span class="gen-progress-timer" id="' + containerId + '-timer">0.0s</span></div>' +
      '<div class="gen-progress-sub" id="' + containerId + '-sub">' + escHtml(stages[0] || '') + '</div>' +
      '<div class="gen-bar-track"><div class="gen-bar-fill"></div></div>' +
      (opts.skeleton ? '<div style="margin-top:14px">' +
        Array(opts.skeleton).fill('<div class="skeleton-row"></div>').join('') + '</div>' : '') +
    '</div>';
  const t0 = performance.now();
  let stageIdx = 0;
  const timer = setInterval(() => {
    const s = ((performance.now() - t0) / 1000);
    const tEl = document.getElementById(containerId + '-timer');
    if (tEl) tEl.textContent = s.toFixed(1) + 's';
    // advance stage roughly every 6s, capped at last stage
    const want = Math.min(stages.length - 1, Math.floor(s / 6));
    if (want !== stageIdx) {
      stageIdx = want;
      const subEl = document.getElementById(containerId + '-sub');
      if (subEl) subEl.textContent = stages[stageIdx];
    }
  }, 100);
  return function stop() {
    clearInterval(timer);
    el.style.display = 'none';
    el.innerHTML = '';
  };
}

async function previewScenarios() {
  const desc = document.getElementById('battery-desc').value.trim();
  const errEl = document.getElementById('preview-error');
  errEl.textContent = '';
  if (!desc) { errEl.textContent = 'Describe what your agent does first.'; return; }

  document.getElementById('preview-btn').disabled = true;
  document.getElementById('understand-btn').disabled = true;
  setStep(3);

  const grounded = (_agentModel && _agentModel.confirmed);
  const groundedId = grounded ? _agentModel.id : null;
  const stop = startProgress('scenario-progress', 'Generating test scenarios',
    grounded
      ? ['Reading your confirmed agent model…',
         'Designing scenarios for each conversation node…',
         'Adding confused, impatient, and adversarial callers…',
         'Injecting audio-stress and edge-case variants…',
         'Finalizing the battery…']
      : ['Analyzing your description…',
         'Designing realistic caller scenarios…',
         'Adding confused, impatient, and adversarial callers…',
         'Injecting audio-stress and edge-case variants…',
         'Finalizing the battery…'],
    {skeleton: 4});

  let resp;
  try {
    resp = await fetch('/battery/preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc, agent_model_id: groundedId}),
    });
  } catch(e) { errEl.textContent = 'Network error: ' + e; stop(); resetPreviewBtn(); return; }

  stop();
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    errEl.textContent = 'Error ' + resp.status + ': ' + (body.detail || resp.statusText);
    resetPreviewBtn();
    return;
  }

  const data = await resp.json();
  _previewScenarios = data.scenarios || [];
  resetPreviewBtn();
  renderScenarioPicker(_previewScenarios);
  document.getElementById('step2').style.display = '';
  if (_agentModel && (_agentModel.nodes || []).length) renderGraphPanel(_agentModel);
  setStep(4);
  document.getElementById('step2').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function resetPreviewBtn() {
  document.getElementById('preview-btn').disabled = false;
  document.getElementById('understand-btn').disabled = false;
}

// ── agent model: understand → confirm ──────────────────────────────────────
let _agentModel = null;

async function understandAgent() {
  const desc = document.getElementById('battery-desc').value.trim();
  const errEl = document.getElementById('preview-error');
  errEl.textContent = '';
  if (!desc) { errEl.textContent = 'Describe what your agent does first.'; return; }

  document.getElementById('understand-btn').disabled = true;
  document.getElementById('preview-btn').disabled = true;
  const stop = startProgress('understand-progress', 'Understanding your agent',
    ['Reading your description…',
     'Extracting goal, required info, and tools…',
     'Inferring business rules and failure risks…',
     'Mapping the conversation graph…'], {});

  let resp;
  try {
    resp = await fetch('/agent-model/extract', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({description: desc}),
    });
  } catch(e) { errEl.textContent = 'Network error: ' + e; stop(); resetPreviewBtn(); return; }

  stop();
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    errEl.textContent = 'Error ' + resp.status + ': ' + (body.detail || resp.statusText);
    resetPreviewBtn();
    return;
  }

  _agentModel = await resp.json();
  resetPreviewBtn();
  renderAgentModel(_agentModel);
  document.getElementById('agent-model').style.display = '';
  setStep(2);
  hydrateIcons();
  document.getElementById('agent-model').scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function renderAgentModel(m) {
  const chips = (arr, cls, get) => (arr || []).length
    ? '<div class="am-chips">' + arr.map(x =>
        '<span class="am-chip ' + (cls||'') + '">' + escHtml(get(x)) + '</span>').join('') + '</div>'
    : '<div class="am-list-item" style="padding-left:0;color:var(--text-3)">None detected</div>';

  const section = (label, inner) =>
    '<div class="am-section"><div class="am-label">' + label + '</div>' + inner + '</div>';

  let html = '';
  html += '<div class="am-goal" style="display:flex;align-items:center;gap:8px"><span style="color:var(--accent)">' + iconSvg('bolt', 16) + '</span>' + escHtml(m.goal || 'No goal detected') + '</div>';

  html += section('Required information',
    chips(m.required_info, '', x => x.label + (x.required ? '' : ' (optional)')));

  html += section('Tools',
    chips(m.tools, '', x => x.name + (x.purpose ? ' — ' + x.purpose : '')));

  html += section('Business rules',
    (m.business_rules || []).length
      ? '<div class="am-chips">' + m.business_rules.map(r =>
          '<span class="am-chip rule">' + escHtml(r.rule) + '</span>').join('') + '</div>'
      : '<div class="am-list-item" style="padding-left:0;color:var(--text-3)">None detected</div>');

  html += section('Detected risks',
    chips(m.risks, 'risk', x => x.description));

  document.getElementById('am-body').innerHTML = html;
  document.getElementById('am-confirmed-badge').textContent = m.confirmed ? 'Confirmed' : 'Draft';
  document.getElementById('am-confirmed-badge').className = 'badge ' + (m.confirmed ? 'pass' : 'pending');

  const graphSection = document.getElementById('node-graph-section');
  if (m.confirmed && (m.nodes || []).length) {
    graphSection.style.display = '';
    renderGraphPanel(m);
  } else {
    graphSection.style.display = 'none';
  }
}

// ── node graph panel: left-to-right flow, clickable nodes ──────────────────
let _selectedNodeId = null;

// Layered left-to-right SVG graph. Nodes are assigned to columns by longest-path
// depth from a root; each column is stacked vertically. Edges are drawn as bezier
// curves with arrowheads and (optional) condition labels.
function renderGraphPanel(m) {
  const nodes = m.nodes || [];
  const edges = (m.edges || []).filter(e =>
    nodes.some(n => n.id === e.source) && nodes.some(n => n.id === e.target));
  const body = document.getElementById('graph-panel-body');
  if (!nodes.length) { body.innerHTML = ''; return; }

  // ── layout dims ──
  const NW = 168, NH = 62, COL_GAP = 84, ROW_GAP = 26, PAD = 24;

  // depth = longest path from any root (nodes with no incoming edge)
  const incoming = {}; nodes.forEach(n => incoming[n.id] = 0);
  edges.forEach(e => { if (e.source !== e.target) incoming[e.target]++; });
  const outAdj = {}; nodes.forEach(n => outAdj[n.id] = []);
  edges.forEach(e => { if (e.source !== e.target) outAdj[e.source].push(e.target); });

  const depth = {};
  const order = nodes.map(n => n.id);
  // seed roots at 0; anything unreferenced also 0
  order.forEach(id => { if (incoming[id] === 0) depth[id] = 0; });
  // relax a few passes (DAG-ish; cycles just settle)
  for (let pass = 0; pass < nodes.length; pass++) {
    let changed = false;
    edges.forEach(e => {
      if (e.source === e.target) return;
      const ds = depth[e.source] == null ? 0 : depth[e.source];
      if (depth[e.target] == null || depth[e.target] < ds + 1) { depth[e.target] = ds + 1; changed = true; }
    });
    if (!changed) break;
  }
  nodes.forEach(n => { if (depth[n.id] == null) depth[n.id] = 0; });

  // group into columns, preserving declaration order within a column
  const cols = {};
  nodes.forEach(n => { (cols[depth[n.id]] = cols[depth[n.id]] || []).push(n); });
  const colKeys = Object.keys(cols).map(Number).sort((a,b)=>a-b);

  // positions
  const pos = {};
  const maxRows = Math.max(...colKeys.map(k => cols[k].length));
  const svgH = PAD*2 + maxRows*NH + (maxRows-1)*ROW_GAP;
  colKeys.forEach((k, ci) => {
    const colNodes = cols[k];
    const colH = colNodes.length*NH + (colNodes.length-1)*ROW_GAP;
    const yStart = PAD + (svgH - PAD*2 - colH)/2;
    colNodes.forEach((n, ri) => {
      pos[n.id] = { x: PAD + ci*(NW+COL_GAP), y: yStart + ri*(NH+ROW_GAP) };
    });
  });
  const svgW = PAD*2 + colKeys.length*NW + (colKeys.length-1)*COL_GAP;

  // scenario coverage per node
  const covById = {};
  (_previewScenarios || []).forEach(s => { if (s.node_id) covById[s.node_id] = (covById[s.node_id]||0)+1; });

  // ── edges ──
  let edgeSvg = '';
  edges.forEach((e, i) => {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) return;
    const x1 = a.x + NW, y1 = a.y + NH/2;
    const x2 = b.x,      y2 = b.y + NH/2;
    const backwards = x2 <= x1;
    const mx = (x1 + x2) / 2;
    let d;
    if (backwards) {
      // route with a downward loop so it reads clearly
      const dip = Math.max(y1, y2) + NH;
      d = 'M'+x1+','+y1+' C'+(x1+50)+','+dip+' '+(x2-50)+','+dip+' '+x2+','+y2;
    } else {
      d = 'M'+x1+','+y1+' C'+mx+','+y1+' '+mx+','+y2+' '+x2+','+y2;
    }
    edgeSvg += '<path class="graph-edge-path" d="'+d+'" marker-end="url(#gm-arrow)"/>';
    if (e.condition) {
      const lx = backwards ? (x1+x2)/2 : mx;
      const ly = backwards ? Math.max(y1,y2)+NH-6 : (y1+y2)/2 - 6;
      const txt = e.condition.toUpperCase();
      const w = txt.length*5.6 + 10;
      edgeSvg += '<rect class="graph-edge-label-bg" x="'+(lx-w/2)+'" y="'+(ly-11)+'" width="'+w+'" height="15" rx="4"/>' +
        '<text class="graph-edge-label" x="'+lx+'" y="'+ly+'" text-anchor="middle">'+escHtml(txt)+'</text>';
    }
  });

  // ── nodes (foreignObject so we keep HTML styling) ──
  let nodeSvg = '';
  nodes.forEach(n => {
    const p = pos[n.id];
    const cov = covById[n.id] || 0;
    const covCls = cov > 0 ? 'covered' : 'empty';
    const covTxt = cov > 0 ? cov + ' scenario' + (cov>1?'s':'') : 'no scenarios';
    nodeSvg +=
      '<foreignObject class="graph-node-fo" x="'+p.x+'" y="'+p.y+'" width="'+NW+'" height="'+NH+'" ' +
        'onclick="showNodePanel(\''+escJs(n.id)+'\')">' +
        '<div xmlns="http://www.w3.org/1999/xhtml" class="graph-node-box'+(n.id===_selectedNodeId?' selected':'')+'" title="'+escHtml(n.objective||'')+'">' +
          '<div class="gn-label">'+escHtml(n.label)+'</div>' +
          '<div class="gn-cov '+covCls+'"><span class="gn-dot"></span>'+covTxt+'</div>' +
        '</div>' +
      '</foreignObject>';
  });

  body.innerHTML =
    '<svg class="graph-svg" width="'+svgW+'" height="'+svgH+'" viewBox="0 0 '+svgW+' '+svgH+'">' +
      '<defs><marker id="gm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
        '<path d="M0,0 L10,5 L0,10 z" fill="#b4b0d8"/></marker></defs>' +
      edgeSvg + nodeSvg +
    '</svg>';
}

function escJs(s) { return String(s).replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

function _scenariosForNode(nodeId) {
  return (_previewScenarios || []).filter(s => s.node_id === nodeId);
}

function showNodePanel(nodeId) {
  if (!_agentModel) return;
  _selectedNodeId = nodeId;
  renderGraphPanel(_agentModel);

  const node = (_agentModel.nodes || []).find(n => n.id === nodeId);
  if (!node) return;

  const scenarios = _scenariosForNode(nodeId);
  const passed = scenarios.filter(s => s.passed === true).length;
  const failed = scenarios.filter(s => s.passed === false).length;

  const rows = scenarios.length
    ? scenarios.map(s =>
        '<div class="node-scenario-row"><span class="badge ' +
        (s.passed === true ? 'pass' : s.passed === false ? 'fail' : 'pending') + '">' +
        (s.passed === true ? 'PASS' : s.passed === false ? 'FAIL' : '—') + '</span>' +
        '<span>' + escHtml(s.title) + '</span></div>'
      ).join('')
    : '<div class="node-scenario-row" style="color:var(--text-3)">No scenarios target this node yet.</div>';

  const html =
    '<button class="node-panel-close" onclick="closeNodePanel()">Close</button>' +
    '<div class="node-panel-title">' + escHtml(node.label) + '</div>' +
    (node.objective ? '<div class="node-panel-objective">' + escHtml(node.objective) + '</div>' : '') +
    '<div class="node-panel-stat-row">' +
      '<div class="node-panel-stat"><div class="stat-num">' + scenarios.length + '</div><div class="stat-label">Scenarios</div></div>' +
      '<div class="node-panel-stat pass"><div class="stat-num">' + passed + '</div><div class="stat-label">Passed</div></div>' +
      '<div class="node-panel-stat fail"><div class="stat-num">' + failed + '</div><div class="stat-label">Failed</div></div>' +
    '</div>' +
    '<div class="node-scenario-list">' + rows + '</div>' +
    '<button id="generate-for-node-btn" onclick="generateForNode(\'' + escJs(nodeId) + '\')">' + iconSvg('sparkles',14) + ' Generate 10 more scenarios for this node</button>' +
    '<span id="generate-for-node-spinner" style="display:none;align-items:center;gap:6px;color:var(--text-2);font-size:13px;font-weight:500"><span class="spinner"></span> Generating…</span>' +
    '<div id="generate-for-node-error"></div>';

  document.getElementById('node-panel-container').innerHTML =
    '<div class="node-panel-overlay" onclick="if(event.target===this) closeNodePanel()">' +
      '<div class="node-panel">' + html + '</div>' +
    '</div>';
}

function closeNodePanel() {
  _selectedNodeId = null;
  document.getElementById('node-panel-container').innerHTML = '';
  if (_agentModel) renderGraphPanel(_agentModel);
}

async function generateForNode(nodeId) {
  const errEl = document.getElementById('generate-for-node-error');
  errEl.textContent = '';
  document.getElementById('generate-for-node-btn').disabled = true;
  document.getElementById('generate-for-node-spinner').style.display = 'flex';

  let resp;
  try {
    resp = await fetch('/battery/generate-for-node', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({agent_model_id: _agentModel.id, node_id: nodeId, count: 10}),
    });
  } catch(e) { errEl.textContent = 'Network error: ' + e; }

  document.getElementById('generate-for-node-spinner').style.display = 'none';
  document.getElementById('generate-for-node-btn').disabled = false;

  if (!resp || !resp.ok) {
    const body = resp ? await resp.json().catch(() => ({})) : {};
    errEl.textContent = 'Error: ' + (body.detail || 'could not generate scenarios');
    return;
  }

  const data = await resp.json();
  const newScenarios = data.scenarios || [];
  _previewScenarios = [..._previewScenarios, ...newScenarios];
  renderScenarioPicker(_previewScenarios);
  document.getElementById('step2').style.display = '';
  renderGraphPanel(_agentModel);
  showNodePanel(nodeId);
}

async function confirmAgentModel() {
  if (!_agentModel || !_agentModel.id) return;
  const errEl = document.getElementById('am-error');
  errEl.textContent = '';
  document.getElementById('am-confirm-btn').disabled = true;
  document.getElementById('am-spinner').style.display = 'flex';

  let resp;
  try {
    resp = await fetch('/agent-model/' + _agentModel.id + '/confirm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: _agentModel}),
    });
  } catch(e) { errEl.textContent = 'Network error: ' + e; }

  document.getElementById('am-spinner').style.display = 'none';
  document.getElementById('am-confirm-btn').disabled = false;

  if (!resp || !resp.ok) {
    const body = resp ? await resp.json().catch(() => ({})) : {};
    errEl.textContent = 'Error: ' + (body.detail || 'could not confirm');
    return;
  }
  _agentModel = await resp.json();
  renderAgentModel(_agentModel);
  // Chain straight into scenario generation — the confirmed model grounds it.
  document.getElementById('node-graph-section').scrollIntoView({behavior:'smooth', block:'nearest'});
  await previewScenarios();
}

function renderScenarioPicker(scenarios) {
  document.getElementById('scenario-count').textContent =
    scenarios.length ? '(' + scenarios.length + ')' : '';
  const byGroup = {};
  scenarios.forEach(s => {
    const g = s.group || 'other';
    (byGroup[g] = byGroup[g] || []).push(s);
  });
  const groupKeys = Object.keys(byGroup).sort(
    (a,b) => (GROUP_ORDER.indexOf(a)+1 || 99) - (GROUP_ORDER.indexOf(b)+1 || 99));

  const html = groupKeys.map(gk => {
    const items = byGroup[gk].map(s =>
      '<label class="scenario-label" style="display:flex;align-items:flex-start;gap:10px;padding:8px 10px;border-radius:6px;cursor:pointer;transition:background .1s">' +
        '<input type="checkbox" checked data-scenario-id="' + escHtml(s.id) + '" style="width:auto;margin-top:2px;accent-color:var(--accent)">' +
        '<div><div style="font-weight:600;font-size:13px;color:var(--text)">' + escHtml(s.title) + '</div>' +
        '<div style="font-size:11px;color:var(--text-2);margin-top:2px">' + escHtml(s.category) +
        (s.node_id ? ' · targets <b>' + escHtml(s.node_id) + '</b>' : '') + '</div></div>' +
      '</label>'
    ).join('');
    return '<div style="margin-bottom:12px">' +
      '<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-3);margin-bottom:4px;padding:0 10px">' +
        escHtml(gk.replace(/_/g,' ')) + ' <span style="font-weight:500">(' + byGroup[gk].length + ')</span></div>' +
      items + '</div>';
  }).join('');
  document.getElementById('scenario-picker').innerHTML = html;
}

async function startBattery() {
  const number = document.getElementById('battery-number').value.trim() || 'loopback';
  const errEl = document.getElementById('battery-error');
  errEl.textContent = '';

  const checked = [...document.querySelectorAll('#scenario-picker input[type=checkbox]:checked')];
  if (checked.length === 0) { errEl.textContent = 'Select at least one scenario to run.'; return; }
  const selectedIds = checked.map(cb => cb.getAttribute('data-scenario-id'));

  document.getElementById('battery-btn').disabled = true;
  document.getElementById('battery-spinner').style.display = 'flex';
  document.getElementById('battery-spinner-text').textContent = 'Starting ' + selectedIds.length + ' scenarios…';
  document.getElementById('report-card').style.display = 'none';
  document.getElementById('group-matrix').innerHTML = '';
  _batteryReportLoaded = false;

  const groundedId = (_agentModel && _agentModel.confirmed) ? _agentModel.id : null;

  let resp;
  try {
    resp = await fetch('/battery', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        description: document.getElementById('battery-desc').value.trim(),
        number,
        selected_scenario_ids: selectedIds,
        agent_model_id: groundedId,
      }),
    });
  } catch(e) { errEl.textContent = 'Network error: ' + e; resetBatteryBtn(); return; }

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    errEl.textContent = 'Error ' + resp.status + ': ' + (body.detail || resp.statusText);
    resetBatteryBtn();
    return;
  }

  const {battery_id, total} = await resp.json();
  document.getElementById('battery-spinner-text').textContent =
    'Running ' + total + ' call' + (total > 1 ? 's' : '') + ' (one at a time)…';
  document.getElementById('run-id-label').textContent = battery_id;
  showStatusArea();
  if (_batteryTimer) clearInterval(_batteryTimer);
  _batteryTimer = setInterval(() => pollBattery(battery_id), 900);
  pollBattery(battery_id);
}

function resetBatteryBtn() {
  document.getElementById('battery-btn').disabled = false;
  document.getElementById('battery-spinner').style.display = 'none';
}

async function pollBattery(batteryId) {
  let data;
  try { data = await fetch('/battery/' + batteryId).then(r => r.json()); }
  catch(e) { return; }
  if (data.detail) return;

  const total = data.total || 0;
  const completed = data.completed || 0;
  const pct = total > 0 ? Math.round(completed / total * 100) : 0;
  document.getElementById('progress-text').textContent =
    cap(data.status) + ' — ' + completed + ' of ' + total + ' calls complete';
  document.getElementById('progress-bar').style.width = pct + '%';
  renderGroupMatrix(data.groups || {});

  if (data.status === 'completed' || data.status === 'failed') {
    clearInterval(_batteryTimer);
    resetBatteryBtn();
    if (data.status === 'failed') {
      document.getElementById('battery-error').textContent =
        'Battery failed — check server logs.';
    }
    if (data.status === 'completed' && !_batteryReportLoaded) {
      _batteryReportLoaded = true;
      await loadBatteryReport(batteryId);
    }
  }
}

function renderGroupMatrix(groups) {
  const keys = Object.keys(groups).sort(
    (a,b) => (GROUP_ORDER.indexOf(a)+1 || 99) - (GROUP_ORDER.indexOf(b)+1 || 99));
  const rows = keys.map(k => {
    const g = groups[k];
    const done = g.passed + g.failed;
    const pills =
      (g.passed ? '<span class="pill-pass">✓ ' + g.passed + '</span>' : '') +
      (g.failed ? '<span class="pill-fail">✗ ' + g.failed + '</span>' : '') +
      (g.running ? '<span class="pill-run">' + g.running + ' running</span>' : '');
    return '<div class="group-row">' +
      '<span class="group-name">' + escHtml(k.replace(/_/g,' ')) + '</span>' +
      '<span class="group-counts">' + (pills || '<span class="group-progress">—</span>') + '</span>' +
      '<span class="group-progress">' + done + '/' + g.total + '</span>' +
    '</div>';
  }).join('');
  document.getElementById('group-matrix').innerHTML =
    '<div class="group-matrix">' + rows + '</div>';
}

// ── report ─────────────────────────────────────────────────────────────────
async function loadBatteryReport(batteryId) {
  let data;
  try { data = await fetch('/battery/' + batteryId + '/report').then(r => r.json()); }
  catch(e) {
    document.getElementById('battery-error').textContent = 'Failed to load report: ' + e;
    return;
  }
  const rep = data.report;
  if (!rep) { document.getElementById('battery-error').textContent = 'Report not ready.'; return; }

  document.getElementById('report-card').style.display = '';

  const pct = rep.total > 0 ? Math.round(rep.passed / rep.total * 100) : 0;
  const verdictMap = {
    production_ready: ['ready', 'Production ready', 'Your agent handled the stress battery well.'],
    needs_work:       ['needs', 'Needs work', 'Your agent passed most tests but has real gaps.'],
    not_ready:        ['notready', 'Not production ready', 'Your agent failed a large share of realistic callers.'],
  };
  const [cls, label, sub] = verdictMap[rep.verdict] || ['needs','Needs work',''];
  document.getElementById('report-verdict').innerHTML =
    '<div class="verdict-banner ' + cls + '">' +
      '<div class="verdict-pct">' + pct + '%</div>' +
      '<div><div class="verdict-label">' + label + '</div>' +
      '<div class="verdict-sub">' + rep.passed + ' of ' + rep.total + ' scenarios passed · ' + escHtml(sub) + '</div></div>' +
    '</div>';

  // Critical failures — grouped by root cause
  const crit = rep.critical_failures || [];
  const byCause = {};
  crit.forEach(f => {
    const key = f.root_cause || 'other';
    (byCause[key] = byCause[key] || []).push(f);
  });
  const causeKeys = Object.keys(byCause).sort(
    (a,b) => causeCount(byCause[b]) - causeCount(byCause[a]));

  function causeCount(fs) { return fs.reduce((n,f) => n + f.affected_count, 0); }
  function causeLabel(key) {
    return key.replace(/_/g,' ').replace(/\\b\\w/g, c => c.toUpperCase());
  }

  document.getElementById('report-critical').innerHTML = crit.length ?
    '<h2>' + crit.length + ' critical failure' + (crit.length>1?'s':'') + '</h2>' +
    causeKeys.map(key => {
      const fs = byCause[key];
      const count = causeCount(fs);
      const fix = fs.find(f => f.suggested_fix)?.suggested_fix || '';
      const id = 'cause-' + key.replace(/[^a-z0-9]/gi,'-');
      return '<details class="root-cause-group">' +
        '<summary><span class="root-cause-title">' + count + ' failure' + (count>1?'s':'') +
          ': ' + escHtml(causeLabel(key)) + '</span>' +
          '<span class="summary-chevron">▼</span></summary>' +
        '<div class="root-cause-body">' +
          (fix ? '<div class="root-cause-fix"><strong>Suggested fix:</strong> ' + escHtml(fix) + '</div>' : '') +
          fs.map(f =>
            '<div class="critical-fail">' +
              '<div class="critical-fail-title">✗ ' + escHtml(f.description) + '</div>' +
              '<div class="critical-fail-meta">Failed in ' + f.affected_count + ' of ' +
                f.total_applicable + ' applicable scenarios</div>' +
              (f.evidence||[]).map(e => '<div class="critical-fail-ev">' + escHtml(e) + '</div>').join('') +
            '</div>'
          ).join('') +
          '<button class="secondary" style="padding:6px 12px;font-size:12px" id="' + id + '" onclick="runRegressionForCause(' +
            JSON.stringify(key) + ',' + JSON.stringify(causeLabel(key)) + ')">' +
            'Run regression test for this cause</button>' +
          '<div class="root-cause-regression-result" id="' + id + '-result"></div>' +
        '</div>' +
      '</details>';
    }).join('')
    : '<h2>No critical failures</h2><div class="dim-reason">No dimension failed in a majority of scenarios.</div>';

  // Graph coverage — one cell per AgentModel node (only present for grounded batteries)
  const graphCov = rep.graph_coverage || [];
  document.getElementById('report-graph-coverage').innerHTML = graphCov.length ?
    '<h2>Graph coverage <span style="text-transform:none;font-weight:500;color:var(--text-3)">· per node, from your confirmed agent model</span></h2>' +
    '<div class="coverage-grid">' + graphCov.map(n => {
      const stat = n.total === 0 ? 'not tested' : n.passed + '/' + n.total + ' passed';
      return '<div class="coverage-cell' + (n.total === 0 ? ' crit' : '') + '">' +
        '<div class="coverage-dim">' + escHtml(n.label) + '</div>' +
        '<div class="coverage-stat">' + stat + '</div>' +
      '</div>';
    }).join('') + '</div>'
    : '';

  // Coverage by dimension
  const dims = rep.dimensions || {};
  const covCells = Object.entries(dims).map(([name, d]) => {
    const applicable = d.passed + d.failed;
    const stat = applicable === 0
      ? 'not tested'
      : d.passed + '/' + applicable + ' passed';
    return '<div class="coverage-cell' + (d.critical ? ' crit' : '') + '">' +
      '<div class="coverage-dim">' + escHtml(DIM_LABELS[name] || name) + '</div>' +
      '<div class="coverage-stat">' + stat + '</div>' +
    '</div>';
  }).join('');
  document.getElementById('report-coverage').innerHTML =
    '<h2>Coverage by dimension <span style="text-transform:none;font-weight:500;color:var(--text-3)">· ' +
    Math.round(rep.coverage_pct) + '% of cells exercised</span></h2>' +
    '<div class="coverage-grid">' + covCells + '</div>';

  // Per-scenario list grouped
  const results = rep.scenario_results || [];
  const byGroup = {};
  results.forEach(r => { (byGroup[r.group||'other'] = byGroup[r.group||'other'] || []).push(r); });
  const groupKeys = Object.keys(byGroup).sort(
    (a,b) => (GROUP_ORDER.indexOf(a)+1 || 99) - (GROUP_ORDER.indexOf(b)+1 || 99));
  document.getElementById('report-scenarios').innerHTML =
    '<h2>All scenarios</h2>' +
    groupKeys.map(gk => {
      const items = byGroup[gk].map(r =>
        '<details class="call-detail"><summary>' +
          '<span class="badge ' + (r.passed?'pass':'fail') + '">' + (r.passed?'PASS':'FAIL') + '</span>' +
          '<span style="font-weight:600">' + escHtml(r.title) + '</span>' +
          '<span class="summary-chevron">▼</span></summary>' +
          '<div class="call-detail-body"><pre style="white-space:pre-wrap;font-family:inherit;font-size:13px;line-height:1.6;color:var(--text-2);margin:0">' +
          escHtml(r.summary || '') + '</pre></div>' +
        '</details>'
      ).join('');
      return '<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--text-3);margin:16px 0 8px">' +
        escHtml(gk.replace(/_/g,' ')) + '</div>' + items;
    }).join('');
}

async function runRegressionForCause(rootCauseKey, rootCauseLabel) {
  const id = 'cause-' + rootCauseKey.replace(/[^a-z0-9]/gi,'-');
  const btn = document.getElementById(id);
  const resultEl = document.getElementById(id + '-result');
  if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
  resultEl.textContent = '';

  const groundedId = (_agentModel && _agentModel.confirmed) ? _agentModel.id : null;
  try {
    const resp = await fetch('/battery/generate-regression', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        agent_model_id: groundedId,
        root_cause_description: rootCauseLabel,
      }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const data = await resp.json();
    resultEl.innerHTML = '<div class="root-cause-regression-list">' +
      'Generated ' + data.scenarios.length + ' regression scenario' +
      (data.scenarios.length>1?'s':'') + ' targeting "' + escHtml(rootCauseLabel) + '":' +
      data.scenarios.map(s => '<div class="critical-fail-ev">' + escHtml(s.title) + '</div>').join('') +
      '</div>';
  } catch (e) {
    resultEl.textContent = 'Failed to generate regression scenarios: ' + e.message;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Run regression test for this cause'; }
  }
}

async function loadBatteryById() {
  const batteryId = document.getElementById('load-battery-id').value.trim();
  const errEl = document.getElementById('load-error-msg');
  errEl.textContent = '';
  if (!batteryId) { errEl.textContent = 'Enter a battery ID.'; return; }
  if (_batteryTimer) { clearInterval(_batteryTimer); _batteryTimer = null; }

  const data = await fetch('/battery/' + batteryId).then(r => r.json()).catch(() => null);
  if (!data || data.detail) { errEl.textContent = 'Battery not found.'; return; }

  document.getElementById('run-id-label').textContent = batteryId;
  showStatusArea();
  document.getElementById('progress-text').textContent =
    cap(data.status) + ' — ' + data.completed + ' of ' + data.total + ' calls';
  document.getElementById('progress-bar').style.width = '100%';
  renderGroupMatrix(data.groups || {});

  if (data.status === 'completed') {
    await loadBatteryReport(batteryId);
  } else if (data.status === 'running' || data.status === 'pending') {
    _batteryReportLoaded = false;
    _batteryTimer = setInterval(() => pollBattery(batteryId), 900);
  } else {
    errEl.textContent = 'Battery status: ' + data.status + ' — no report available.';
  }
}

// ── Production Learning ─────────────────────────────────────────────────────

function _parseProductionCallsInput(raw) {
  raw = raw.trim();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.map(c => ({
        transcript: c.transcript || '',
        outcome: c.outcome || 'error',
        error: c.error ?? null,
      })).filter(c => c.transcript.trim());
    }
  } catch (e) { /* fall through to plain-text parsing */ }

  return raw.split('\n').map(l => l.trim()).filter(Boolean).map(line => ({
    transcript: line,
    outcome: 'error',
    error: null,
  }));
}

async function ingestProductionCalls() {
  const raw = document.getElementById('production-calls-input').value;
  const errEl = document.getElementById('ingest-error');
  const resultEl = document.getElementById('ingest-result');
  errEl.textContent = '';
  resultEl.textContent = '';

  const calls = _parseProductionCallsInput(raw);
  if (calls.length === 0) { errEl.textContent = 'Paste at least one call (JSON array or plain text, one transcript per line).'; return; }

  document.getElementById('ingest-btn').disabled = true;
  document.getElementById('ingest-spinner').style.display = 'flex';

  let resp;
  try {
    resp = await fetch('/production/ingest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({calls}),
    });
  } catch(e) {
    errEl.textContent = 'Network error: ' + e;
    document.getElementById('ingest-btn').disabled = false;
    document.getElementById('ingest-spinner').style.display = 'none';
    return;
  }

  document.getElementById('ingest-btn').disabled = false;
  document.getElementById('ingest-spinner').style.display = 'none';

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    errEl.textContent = 'Error ' + resp.status + ': ' + (body.detail || resp.statusText);
    return;
  }

  const data = await resp.json();
  resultEl.textContent = data.new_failures + ' new failure(s) detected — ' +
    data.regression_scenarios_generated + ' regression scenario(s) generated.';
}

async function loadProductionFailures() {
  const listEl = document.getElementById('production-failures-list');
  const bodyEl = document.getElementById('production-failures-body');
  const data = await fetch('/production/failures').then(r => r.json()).catch(() => null);
  listEl.style.display = 'block';
  if (!data || !data.failures) { bodyEl.innerHTML = '<div class="error-msg">Could not load failures.</div>'; return; }
  if (data.failures.length === 0) { bodyEl.innerHTML = '<div class="card-subtitle">No production failures recorded yet.</div>'; return; }

  bodyEl.innerHTML = data.failures.map(f => `
    <div class="node-scenario-row" style="align-items:flex-start;flex-direction:column;gap:4px;margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;width:100%;gap:8px">
        <span class="badge fail">${escHtml(f.root_cause || 'other')}</span>
        <span style="color:var(--text-3);font-weight:500">${escHtml(f.detected_at)}</span>
      </div>
      <div style="font-weight:500">${escHtml(f.transcript.slice(0, 200))}${f.transcript.length > 200 ? '…' : ''}</div>
      <button class="secondary" style="padding:4px 10px;font-size:11px" onclick="generateScenarioFromFailure('${f.id}')">Generate regression scenario</button>
    </div>
  `).join('');
}

async function generateScenarioFromFailure(callId) {
  const resp = await fetch('/production/failures/generate-scenario', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({call_id: callId}),
  });
  if (!resp.ok) { alert('Failed to generate scenario.'); return; }
  const data = await resp.json();
  alert('Generated ' + data.scenarios.length + ' regression scenario(s).');
}

function cap(s) { return String(s||'').charAt(0).toUpperCase() + String(s||'').slice(1); }

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    # no-store so the browser never serves a stale copy of the single-page app
    # (uvicorn --reload restarts Python, but won't bust a cached HTML page).
    return HTMLResponse(
        _PAGE_HTML,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )

