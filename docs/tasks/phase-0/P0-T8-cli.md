# P0-T8 — CLI: run one call end-to-end → report

**Context pack:** `tech-stack.md` + `conventions.md` + this file.
**Depends on:** P0-T1..T7

## Goal
A single command that ties the engine + scorer together and delivers the Phase 0 magic
moment from the terminal.

## Deliverables
- `voicetest/cli.py` (argparse/typer): `python -m voicetest.cli run`
  - `--scenario <id-or-path>` (default: the builtin appointment scenario)
  - `--number <e164>` → uses `TwilioOutbound` (real call)
  - `--loopback` → uses `LocalLoopback` (no telephony; for demos/dev)
  - `--broken` → LocalLoopback with the broken-agent fixture (proves scorer fails)
  - flow: load scenario → pick target → `run_call` → `score` → write
    `report.json` + recording → print a human summary (pass/fail, reasons, p50/p95,
    paths).
- A `list` subcommand: `python -m voicetest.cli list` shows builtin scenarios.

## Tests to write (integration, no network)
- `test_cli_loopback.py`: `run --loopback` exits 0, writes report.json with a `passed`
  field and a recording path.
- `test_cli_broken.py`: `run --broken` produces `passed=False` with a sensible reason
  (the scorer-trust gate from testing-strategy.md).

## Definition of done — **and Phase 0 done**
- [ ] `run --loopback` works offline end-to-end and prints a report
- [ ] `run --broken` correctly fails (scorer trust established)
- [ ] `run --number <your verified #>` places a real call and returns a recording +
      scored report (manual smoke)
- [ ] You can show someone: "I gave it a number, it acted like a difficult caller, and
      here's the recording + score."
