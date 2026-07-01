"""
jobs.py — asyncio-based job runner for voicetest runs.

A *run* = N calls of a given scenario against a phone number.

Phase 1 executes calls sequentially (the outer loop is a plain for loop).
The design is ready for asyncio.gather + semaphore in Phase 2: run_call and
score are already async-pure functions; only the orchestration loop changes.

Job lifecycle
-------------
PENDING → RUNNING → COMPLETED (or FAILED on unhandled runner error)

Each call:
1. create_call row (store)
2. await run_call(scenario, target)
3. await score(result, scenario, judge)
4. update_call row + save_report row (store)
5. increment completed count on the run
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from voicetest.config import Settings

if TYPE_CHECKING:
    from voicetest.agent_model.models import AgentModel
from voicetest.engine.pipeline import Providers, real_providers
from voicetest.log import vlog
from voicetest.engine.run_call import run_call
from voicetest.models import CallOutcome, CallResult, Report
from voicetest.providers.targets import CallTarget, LocalLoopback
from voicetest.providers.telephony import is_loopback
from voicetest.scenarios.loader import load_all
from voicetest.scenarios.models import Scenario
from voicetest.scoring.judge import JudgeProvider
from voicetest.scoring.score import score
from voicetest.store import RunStatus, Store


def _make_target(number: str, settings: Settings | None = None) -> CallTarget:
    """
    Resolve a phone number to a CallTarget — the single dispatch point.

    Loopback / +0… numbers → in-process LocalLoopback (no telephony, no cost).
    Any real number → TwilioOutbound, which places a real outbound call and
    bridges audio through the Media Streams WS server.
    """
    if is_loopback(number):
        return LocalLoopback()
    from voicetest.providers.twilio_target import TwilioOutbound

    return TwilioOutbound(number, settings=settings)


def _make_providers(number: str, settings: Settings | None = None) -> Providers | None:
    """
    Choose the provider bundle for a call.

    Loopback uses the stub providers (the LocalLoopback stub agent + text frames).
    A real call needs real STT/TTS/LLM so actual phone audio is transcribed,
    answered, and synthesized. Returning None lets run_call use its stub default.
    """
    if is_loopback(number):
        return None
    settings = settings or Settings()
    from voicetest.providers.llm import make_llm_provider
    from voicetest.providers.stt import DeepgramSTT
    from voicetest.providers.tts import CartesiaTTS

    if not settings.deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY required to place a real call.")
    if not settings.cartesia_api_key:
        raise ValueError("CARTESIA_API_KEY required to place a real call.")

    return real_providers(
        stt=DeepgramSTT(api_key=settings.deepgram_api_key),
        tts=CartesiaTTS(api_key=settings.cartesia_api_key),
        llm=make_llm_provider(settings.llm_caller_model),
    )


async def execute_run(
    run_id: str,
    scenario: Scenario,
    number: str,
    n: int,
    store: Store,
    judge: JudgeProvider | None = None,
    target_factory: Callable[[str], CallTarget] | None = None,
) -> None:
    """
    Execute N calls for *scenario* against *number* and persist results.

    Designed to run as an asyncio background task.  Errors within individual
    calls are captured in the call row; only a catastrophic runner error
    (e.g. broken store) flips the run to FAILED.
    """
    make_target = target_factory or _make_target
    vlog("job", f"run started run={run_id[:8]} scenario={scenario.id} number={number} n={n}")
    store.update_run_status(run_id, RunStatus.RUNNING, completed=0)

    call_id: str | None = None
    completed = 0
    try:
        for i in range(n):
            call_id = str(uuid.uuid4())
            vlog("job", f"call {i + 1}/{n} starting (call_id={call_id[:8]})")
            await asyncio.to_thread(store.create_call, call_id, run_id, i)

            try:
                target = make_target(number)
                providers = _make_providers(number)
            except Exception as exc:  # noqa: BLE001
                vlog("job", f"!!! target/provider SETUP failed: {type(exc).__name__}: {exc}")
                # Target/provider setup failed (e.g. telephony not configured, a
                # non-allowlisted number, or a missing API key). Record it on the
                # call row so the UI can show *why*, then mark the run failed
                # instead of dying silently.
                result = CallResult(
                    scenario_id=scenario.id,
                    outcome=CallOutcome.ERROR,
                    turns=[],
                    events=[],
                    recording_path=None,
                    duration_ms=0,
                    error=str(exc),
                )
                await asyncio.to_thread(store.update_call, call_id, result)
                await asyncio.to_thread(
                    store.update_run_status, run_id, RunStatus.FAILED, i
                )
                return

            result = await run_call(scenario, target, providers)
            vlog("job", f"call {i + 1}/{n} finished outcome={result.outcome} "
                        f"turns={len(result.turns)} dur={result.duration_ms}ms "
                        f"error={result.error!r}")

            await asyncio.to_thread(store.update_call, call_id, result)

            report = await score(result, scenario, judge)
            report_id = str(uuid.uuid4())
            await asyncio.to_thread(store.save_report, report_id, call_id, run_id, report)

            completed = i + 1
            await asyncio.to_thread(
                store.update_run_status, run_id, RunStatus.RUNNING, completed
            )

        vlog("job", f"run COMPLETED run={run_id[:8]} ({n} calls)")
        await asyncio.to_thread(store.update_run_status, run_id, RunStatus.COMPLETED, n)

    except Exception as exc:  # noqa: BLE001
        vlog("job", f"!!! run FAILED run={run_id[:8]}: {type(exc).__name__}: {exc}")
        # Mark run as failed so callers don't poll forever. Record the error on
        # the in-flight call row so the UI can surface *why* (e.g. a missing
        # provider/API key at the scoring step), then re-raise so the task's
        # exception is still visible via Task.exception().
        try:
            if call_id is not None:
                err_result = CallResult(
                    scenario_id=scenario.id,
                    outcome=CallOutcome.ERROR,
                    turns=[],
                    events=[],
                    recording_path=None,
                    duration_ms=0,
                    error=str(exc),
                )
                await asyncio.to_thread(store.update_call, call_id, err_result)
            await asyncio.to_thread(
                store.update_run_status, run_id, RunStatus.FAILED, completed
            )
        except Exception:  # noqa: BLE001
            pass
        raise


# ---------------------------------------------------------------------------
# Battery — run a full auto-generated stress-test suite in parallel
# ---------------------------------------------------------------------------


async def _run_one_battery_scenario(
    battery_id: str,
    scenario: Scenario,
    number: str,
    store: Store,
    judge: JudgeProvider | None,
    make_target: Callable[[str], CallTarget],
    sem: asyncio.Semaphore,
) -> tuple[Scenario, Report] | None:
    """Execute a single scenario (one call) inside a battery slot."""
    from voicetest.scoring.score import score

    async with sem:
        run_id = str(uuid.uuid4())
        await asyncio.to_thread(
            store.create_run,
            run_id,
            scenario.id,
            number,
            1,
            battery_id=battery_id,
            scenario_group=scenario.group,
            scenario_title=scenario.title,
        )
        call_id = str(uuid.uuid4())
        await asyncio.to_thread(store.create_call, call_id, run_id, 0)

        try:
            target = make_target(number)
            providers = _make_providers(number)
            result = await run_call(scenario, target, providers)
            await asyncio.to_thread(store.update_call, call_id, result)

            report = await score(result, scenario, judge)
            report_id = str(uuid.uuid4())
            await asyncio.to_thread(
                store.save_report, report_id, call_id, run_id, report
            )
            await asyncio.to_thread(
                store.update_run_status, run_id, RunStatus.COMPLETED, 1
            )
            return scenario, report
        except Exception as exc:  # noqa: BLE001
            vlog("battery", f"scenario {scenario.id} failed: {type(exc).__name__}: {exc}")
            err = CallResult(
                scenario_id=scenario.id,
                outcome=CallOutcome.ERROR,
                turns=[],
                events=[],
                recording_path=None,
                duration_ms=0,
                error=str(exc),
            )
            await asyncio.to_thread(store.update_call, call_id, err)
            await asyncio.to_thread(
                store.update_run_status, run_id, RunStatus.FAILED, 0
            )
            return None
        finally:
            await asyncio.to_thread(store.increment_battery_completed, battery_id)


async def execute_battery(
    battery_id: str,
    scenarios: list[Scenario],
    number: str,
    store: Store,
    judge: JudgeProvider | None = None,
    concurrency: int = 1,
    between_calls_delay_s: float = 1.0,
    target_factory: Callable[[str], CallTarget] | None = None,
    agent_model: "AgentModel | None" = None,
) -> None:
    """
    Run every scenario in *scenarios* sequentially (one at a time), then aggregate.

    `agent_model` is the confirmed AgentModel this battery was generated from
    (if any) — passed through to aggregate_battery so the report includes
    graph coverage (pass/fail per node), not just dimension coverage.
    """
    from voicetest.scoring.aggregate import aggregate_battery
    from voicetest.store import _to_json_safe

    make_target = target_factory or _make_target
    vlog("battery", f"battery {battery_id[:8]} started: {len(scenarios)} scenarios n={number} (sequential)")
    store.update_battery_status(battery_id, RunStatus.RUNNING, 0)

    sem = asyncio.Semaphore(1)  # always sequential
    results = []
    try:
        for i, scenario in enumerate(scenarios):
            vlog("battery", f"scenario {i+1}/{len(scenarios)}: {scenario.id}")
            result = await _run_one_battery_scenario(
                battery_id, scenario, number, store, judge, make_target, sem
            )
            results.append(result)
            if i < len(scenarios) - 1:
                await asyncio.sleep(between_calls_delay_s)

        pairs: list[tuple[Scenario, Report]] = [r for r in results if r is not None]

        report = aggregate_battery(battery_id, pairs, agent_model=agent_model)
        report_json = json.dumps(_to_json_safe(report))
        store.update_battery_status(
            battery_id, RunStatus.COMPLETED, len(scenarios), report_json
        )
        vlog("battery", f"battery {battery_id[:8]} COMPLETED verdict={report.verdict}")
    except Exception as exc:  # noqa: BLE001
        vlog("battery", f"!!! battery {battery_id[:8]} FAILED: {type(exc).__name__}: {exc}")
        try:
            store.update_battery_status(battery_id, RunStatus.FAILED, 0)
        except Exception:  # noqa: BLE001
            pass
        raise


# ---------------------------------------------------------------------------
# In-process task registry (no Redis/Celery — per ADR/roadmap)
# ---------------------------------------------------------------------------

_tasks: dict[str, asyncio.Task] = {}


def start_run(
    run_id: str,
    scenario: Scenario,
    number: str,
    n: int,
    store: Store,
    judge: JudgeProvider | None = None,
    target_factory: Callable[[str], CallTarget] | None = None,
) -> asyncio.Task:
    """
    Schedule execute_run as a background asyncio task and return the Task.

    The task is tracked in _tasks so it isn't garbage-collected while running.
    The entry is cleaned up automatically when the task finishes.
    """
    task = asyncio.create_task(
        execute_run(run_id, scenario, number, n, store, judge, target_factory),
        name=f"run-{run_id}",
    )
    _tasks[run_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        _tasks.pop(run_id, None)

    task.add_done_callback(_cleanup)
    return task


def start_battery(
    battery_id: str,
    scenarios: list[Scenario],
    number: str,
    store: Store,
    judge: JudgeProvider | None = None,
    concurrency: int = 1,
    between_calls_delay_s: float = 1.0,
    target_factory: Callable[[str], CallTarget] | None = None,
    agent_model: "AgentModel | None" = None,
) -> asyncio.Task:
    """Schedule execute_battery as a tracked background task."""
    task = asyncio.create_task(
        execute_battery(
            battery_id, scenarios, number, store, judge, concurrency,
            between_calls_delay_s, target_factory, agent_model=agent_model,
        ),
        name=f"battery-{battery_id}",
    )
    _tasks[battery_id] = task

    def _cleanup(t: asyncio.Task) -> None:
        _tasks.pop(battery_id, None)

    task.add_done_callback(_cleanup)
    return task


def get_builtin_scenarios() -> list[Scenario]:
    """Return all builtin scenarios (used by GET /scenarios)."""
    return load_all()
