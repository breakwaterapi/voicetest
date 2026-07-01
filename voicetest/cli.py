"""
voicetest CLI — Phase 0 entry point.

Usage::
    python -m voicetest.cli run --loopback
    python -m voicetest.cli run --broken
    python -m voicetest.cli run --loopback --scenario appointment-booking-confused-date
    python -m voicetest.cli run --number +15555550100   # real Twilio call (manual smoke)
    python -m voicetest.cli list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print(msg: str) -> None:
    print(msg, flush=True)


def _resolve_scenario(scenario_arg: str | None):
    """Return a Scenario from an id string, file path, or None (builtin default)."""
    from voicetest.scenarios import BOOKING_SCENARIO, load_all, load_scenario
    from voicetest.scenarios.loader import BUILTIN_DIR

    if scenario_arg is None:
        return BOOKING_SCENARIO

    # Try as a file path first
    p = Path(scenario_arg)
    if p.exists():
        return load_scenario(p)

    # Try as a builtin id
    all_scenarios = load_all(BUILTIN_DIR)
    for s in all_scenarios:
        if s.id == scenario_arg:
            return s

    _print(f"Error: scenario '{scenario_arg}' not found as a file path or builtin id.")
    _print("Run `python -m voicetest.cli list` to see available scenarios.")
    sys.exit(1)


def _print_report_summary(report, report_json_path: Path | None) -> None:
    """Pretty-print a scored report to stdout."""
    verdict = "PASSED ✓" if report.passed else "FAILED ✗"
    _print(f"\n{'='*60}")
    _print(f"  {verdict}")
    _print(f"{'='*60}")
    _print(f"  Scenario : {report.call.scenario_id}")
    _print(f"  Outcome  : {report.call.outcome}")
    _print(f"  Turns    : {len(report.call.turns)}")
    _print(f"  Duration : {report.call.duration_ms}ms")

    if "latency" in report.dimensions:
        lat = report.dimensions["latency"]
        warn = " ⚠" if lat.get("warn") else ""
        _print(f"  Latency  : p50={lat['p50_ms']}ms  p95={lat['p95_ms']}ms{warn}")

    if "goal_completion" in report.dimensions:
        goal = report.dimensions["goal_completion"]
        _print(f"\n  Goal completed : {'Yes' if goal['goal_completed'] else 'No'}")
        _print(f"  Reason         : {goal['reason']}")
        if goal.get("criteria"):
            _print("\n  Pass criteria:")
            for c in goal["criteria"]:
                tick = "✓" if c["met"] else "✗"
                _print(f"    {tick} {c['text']}")
    else:
        _print(f"\n  Summary: {report.summary}")

    if report.call.recording_path:
        _print(f"\n  Recording : {report.call.recording_path}")
    if report_json_path:
        _print(f"  Report    : {report_json_path}")
    _print("")


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------


async def _run_async(args: argparse.Namespace) -> int:
    from voicetest.engine.run_call import run_call
    from voicetest.scoring.score import score
    from voicetest.store import BlobStore

    scenario = _resolve_scenario(args.scenario)

    # --- Pick target ---
    if args.number:
        try:
            from voicetest.providers.twilio_target import TwilioOutbound
        except ImportError:
            _print("Error: TwilioOutbound not available (P0-T7 not yet implemented).")
            return 1
        target = TwilioOutbound(number=args.number)
        providers = None  # use real providers from config
        judge = None      # use real judge from config
    elif args.broken:
        from tests.fixtures.broken_agent import make_broken_agent
        from tests.fixtures import stub_providers
        from voicetest.providers.targets import LocalLoopback
        from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult

        target = LocalLoopback(agent_fn=make_broken_agent(delay_s=0.05))
        providers = stub_providers(responses=[
            "Hi, I'd like to book a table please.",
            "Next Saturday at 7pm for two people.",
            "Yes, 617-555-0142. Thank you, goodbye!",
        ])
        # CannedJudge that reflects what a broken agent produces: goal not met
        canned = JudgeResult(
            goal_completed=False,
            reason="The agent failed to confirm any booking details.",
            criteria=[
                CriterionResult(
                    text=c,
                    met=False,
                    evidence="not found in transcript",
                )
                for c in scenario.scoring.pass_criteria
            ],
        )
        judge = CannedJudge(canned)
    else:
        # --loopback (default)
        from tests.fixtures import stub_providers
        from voicetest.providers.targets import LocalLoopback
        from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult

        target = LocalLoopback()
        providers = stub_providers(responses=[
            "Hi, I'd like to book a haircut for next Saturday afternoon.",
            "Afternoon, around 2pm. My number is 617-555-0142.",
            "Yes that's correct. Thank you, goodbye!",
        ])
        canned = JudgeResult(
            goal_completed=True,
            reason="The agent confirmed next Saturday afternoon and the phone number.",
            criteria=[
                CriterionResult(text=c, met=True, evidence="confirmed in transcript")
                for c in scenario.scoring.pass_criteria
            ],
        )
        judge = CannedJudge(canned)

    _print(f"Running scenario: {scenario.id}")
    mode = "loopback" if not args.number else f"TwilioOutbound → {args.number}"
    if args.broken:
        mode = "loopback (broken agent)"
    _print(f"Target: {mode}")
    _print("Calling…")

    # --- Execute call ---
    call_result = await run_call(scenario, target, providers)

    _print(f"Call finished. Outcome: {call_result.outcome}  Turns: {len(call_result.turns)}")

    # --- Persist artifacts ---
    blob_dir = Path(args.output) if args.output else None
    store = BlobStore(root_dir=blob_dir)
    report_json_path = store.write_result(call_result)

    _print("Scoring…")
    report = await score(call_result, scenario, judge=judge)

    # Write full report JSON (overwrite with scored version)
    from voicetest.store import call_result_to_dict
    full = {
        "passed": report.passed,
        "summary": report.summary,
        "dimensions": report.dimensions,
        "call": call_result_to_dict(call_result),
    }
    report_json_path.write_text(json.dumps(full, indent=2))

    _print_report_summary(report, report_json_path)
    return 0 if report.passed else 1


def _cmd_run(args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args))


# ---------------------------------------------------------------------------
# test subcommand — describe an agent, generate + run a full battery
# ---------------------------------------------------------------------------


def _print_battery_report(report) -> None:
    """Pretty-print a BatteryReport to stdout."""
    verdict_label = {
        "production_ready": "PRODUCTION READY ✓",
        "needs_work": "NEEDS WORK",
        "not_ready": "NOT PRODUCTION READY ✗",
    }.get(report.verdict, report.verdict)
    pct = round(report.passed / report.total * 100) if report.total else 0

    _print(f"\n{'='*64}")
    _print(f"  {verdict_label}  —  {pct}%  ({report.passed}/{report.total} scenarios passed)")
    _print(f"{'='*64}")

    if report.critical_failures:
        _print(f"\n  {len(report.critical_failures)} critical failure(s):")
        for f in report.critical_failures:
            _print(f"    ✗ {f.description} — failed {f.affected_count}/{f.total_applicable}")
            for ev in f.evidence[:2]:
                _print(f"        · {ev[:90]}")
    else:
        _print("\n  No critical failures.")

    _print("\n  Coverage by dimension:")
    for name, d in report.dimensions.items():
        applicable = d.passed + d.failed
        stat = "not tested" if applicable == 0 else f"{d.passed}/{applicable} passed"
        flag = "  ⚠ CRITICAL" if d.critical else ""
        _print(f"    {name:<20} {stat}{flag}")

    _print("\n  Scenarios:")
    for r in report.scenario_results:
        tick = "✓" if r["passed"] else "✗"
        _print(f"    {tick} [{r['group']}] {r['title']}")
    _print("")


async def _test_async(args: argparse.Namespace) -> int:
    import uuid

    from voicetest.jobs import execute_battery
    from voicetest.store import Store

    number = args.number or "loopback"

    # Pick a compiler: stub for loopback (no API key), real otherwise.
    if number == "loopback" and not args.real_compiler:
        from voicetest.scenarios.compiler import StubScenarioCompiler
        compiler = StubScenarioCompiler()
    else:
        from voicetest.scenarios.compiler import make_compiler
        compiler = make_compiler()

    _print(f"Generating stress-test battery for: {args.description!r}")
    scenarios = await compiler.compile_battery(args.description, args.known or "")
    _print(f"Generated {len(scenarios)} scenarios across "
           f"{len({s.group for s in scenarios})} groups.")

    # Use an isolated temp DB unless one is configured.
    store = Store()
    battery_id = str(uuid.uuid4())
    store.create_battery(battery_id, args.description, number, len(scenarios))

    judge = None
    if number == "loopback" and not args.real_compiler:
        from voicetest.scoring.judge import CannedJudge, JudgeResult
        judge = CannedJudge(JudgeResult(goal_completed=True, reason="loopback dry-run", criteria=[]))

    _print(f"Running {len(scenarios)} calls against {number}…")
    await execute_battery(battery_id, scenarios, number, store, judge=judge)

    battery = store.get_battery(battery_id)
    import json as _json
    rep = _json.loads(battery["report_json"]) if battery and battery.get("report_json") else None
    if rep is None:
        _print("Battery did not produce a report.")
        return 1

    # Reconstruct a lightweight object for printing.
    from voicetest.models import BatteryReport, CriticalFailure, DimensionSummary
    report = BatteryReport(
        battery_id=rep["battery_id"],
        total=rep["total"],
        passed=rep["passed"],
        coverage_pct=rep["coverage_pct"],
        verdict=rep["verdict"],
        dimensions={k: DimensionSummary(**v) for k, v in rep["dimensions"].items()},
        critical_failures=[CriticalFailure(**f) for f in rep["critical_failures"]],
        scenario_results=rep["scenario_results"],
    )
    _print_battery_report(report)
    _print(f"  Battery ID: {battery_id}")
    return 0 if report.verdict == "production_ready" else 1


def _cmd_test(args: argparse.Namespace) -> int:
    return asyncio.run(_test_async(args))


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------


def _cmd_list(_args: argparse.Namespace) -> int:
    from voicetest.scenarios import load_all
    from voicetest.scenarios.loader import BUILTIN_DIR

    scenarios = load_all(BUILTIN_DIR)
    if not scenarios:
        _print("No builtin scenarios found.")
        return 0

    _print(f"\nBuiltin scenarios ({len(scenarios)}):")
    _print(f"  {'ID':<45} {'CATEGORY':<25} TITLE")
    _print("  " + "-" * 90)
    for s in scenarios:
        _print(f"  {s.id:<45} {s.category:<25} {s.title}")
    _print("")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voicetest.cli",
        description="VoiceTest — synthetic caller engine CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run one synthetic call and score it")
    target_group = run_p.add_mutually_exclusive_group()
    target_group.add_argument(
        "--loopback",
        action="store_true",
        default=False,
        help="Use LocalLoopback (no telephony; default when no --number given)",
    )
    target_group.add_argument(
        "--broken",
        action="store_true",
        default=False,
        help="Use LocalLoopback with the broken-agent fixture (should score as FAILED)",
    )
    target_group.add_argument(
        "--number",
        metavar="E164",
        default=None,
        help="Dial this E.164 number via TwilioOutbound (requires Twilio env vars)",
    )
    run_p.add_argument(
        "--scenario",
        metavar="ID_OR_PATH",
        default=None,
        help="Scenario id (builtin) or path to a YAML file (default: appointment-booking-confused-date)",
    )
    run_p.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Directory to write artifacts (default: VOICETEST_BLOB_DIR from config)",
    )

    # --- test (describe → battery) ---
    test_p = sub.add_parser(
        "test",
        help="Describe an agent and run a full auto-generated stress-test battery",
    )
    test_p.add_argument(
        "description",
        help="Plain-English description of what your agent does",
    )
    test_p.add_argument(
        "--number",
        metavar="E164",
        default=None,
        help="E.164 number to dial via Twilio (default: loopback dry-run)",
    )
    test_p.add_argument(
        "--known",
        metavar="TEXT",
        default=None,
        help="Known issues — generates targeted regression scenarios",
    )
    test_p.add_argument(
        "--real-compiler",
        action="store_true",
        default=False,
        help="Use the real LLM compiler even in loopback (requires API key)",
    )

    # --- list ---
    sub.add_parser("list", help="List builtin scenarios")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        # Default to --loopback if no target flag given
        if not args.number and not args.broken:
            args.loopback = True
        return _cmd_run(args)

    if args.command == "test":
        return _cmd_test(args)

    if args.command == "list":
        return _cmd_list(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
