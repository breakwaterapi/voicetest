"""
ScenarioCompiler — turn a natural-language description into a validated Scenario suite.

Architecture
------------
ScenarioCompiler (Protocol)
    compile(description) -> list[Scenario]

Implementations
    LLMScenarioCompiler  — calls Claude/GPT to generate 3 difficulty variants
    StubScenarioCompiler — returns fixed scenarios for tests (no network)

The compiler output is always plain list[Scenario] — identical to hand-written
YAML scenarios at runtime. No special handling needed downstream.

Compiler version
----------------
All generated scenarios carry compiler_version=1 so reports can distinguish
hand-written from auto-generated test suites.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from voicetest.scenarios.models import (
    BargeInConfig,
    Limits,
    NoiseConfig,
    Persona,
    Scenario,
    ScoringConfig,
)

if TYPE_CHECKING:
    from voicetest.agent_model.models import AgentModel

COMPILER_VERSION = 1

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ScenarioCompiler(Protocol):
    async def compile(self, description: str) -> list[Scenario]:
        """
        Turn a plain-English description of an agent into a suite of Scenarios.

        Returns 3 variants ordered easy → medium → hard.
        """
        ...

    async def compile_battery(
        self,
        description: str,
        known_issues: str = "",
        agent_model: "AgentModel | None" = None,
        target_node_id: str = "",
        count: int = 0,
    ) -> list[Scenario]:
        """
        Turn a description into a full stress-test battery (~20+ scenarios)
        organised into behavioural groups. Each Scenario carries its group.

        When `agent_model` is provided (a confirmed AgentModel), generation is
        GROUNDED in it: the prompt references its real nodes/tools/rules/risks
        instead of re-inferring them from raw text, and each generated scenario
        is tagged with the node_id it targets — the basis for graph coverage.

        When `target_node_id` is also set, generation is FOCUSED on that single
        node: it produces `count` scenarios (default 10) that all target it,
        instead of spreading scenarios across the whole graph.
        """
        ...

    async def compile_regression(
        self, root_cause_description: str
    ) -> list[Scenario]:
        """
        Generate 5 scenarios specifically designed to trigger the failure mode
        described by `root_cause_description` (e.g. "Calendar API timeout").
        """
        ...

    async def compile_from_failure_text(self, transcript: str) -> list[Scenario]:
        """
        Generate 3 targeted regression scenarios from a raw production call
        transcript that failed — used by the production-learning loop to turn
        an ingested failure directly into test coverage.
        """
        ...


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert voice-AI quality engineer. You generate adversarial test
scenarios for voice agents (phone bots). Given a plain-English description of
what a voice agent does, you produce a JSON test suite that exposes realistic
failure modes real callers would trigger.

Return ONLY a raw JSON object — no markdown fences, no prose outside the JSON.
"""

_USER_TEMPLATE = """\
AGENT DESCRIPTION:
{description}

Generate a JSON object with exactly this shape:
{{
  "category": "<snake_case label, e.g. appointment_booking>",
  "variants": [
    {{
      "difficulty": "easy",
      "title": "<short title>",
      "persona_system_prompt": "<2-3 sentences: who is this caller, their communication style>",
      "goal": "<what the caller is trying to accomplish, one sentence>",
      "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
      "ground_truth": {{"<key>": "<expected value>", ...}},
      "objective": "<what the agent must correctly do to pass, one sentence>",
      "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
      "barge_in_enabled": false,
      "barge_in_caller_says": "",
      "noise_enabled": false
    }},
    {{
      "difficulty": "medium",
      ...same shape, but with at least one date/time ambiguity or mid-call correction...
    }},
    {{
      "difficulty": "hard",
      ...same shape, barge_in_enabled: true, noise_enabled: true where sensible,
         caller changes mind or gives conflicting info...
    }}
  ]
}}

Rules:
- difficult_behaviors must have 3–5 items each and be specific and realistic
- ground_truth keys must match what pass_criteria check (e.g. "phone", "date", "time")
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite")
- For "hard": set barge_in_enabled=true and provide a barge_in_caller_says string
- For "hard" with voice noise context: set noise_enabled=true
- Keep persona_system_prompt ≤ 3 sentences — it goes in a TTS caller's system prompt
- Tailor behaviors to the specific domain (dental → insurance/timing; real estate → budget/location)
"""


_BATTERY_USER_TEMPLATE = """\
AGENT DESCRIPTION:
{description}
{known_issues_block}
Generate a comprehensive stress-test BATTERY for this voice agent. Produce a
JSON object with this exact shape:
{{
  "category": "<snake_case label, e.g. appointment_booking>",
  "groups": [
    {{
      "group": "<one of: baseline | confused_caller | impatient_caller | edge_cases | audio_stress | adversarial | regression>",
      "scenarios": [
        {{
          "title": "<short title>",
          "persona_system_prompt": "<2-3 sentences: who this caller is, their style>",
          "goal": "<what the caller wants, one sentence>",
          "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
          "ground_truth": {{"<key>": "<expected value>", ...}},
          "objective": "<what the agent must correctly do to pass, one sentence>",
          "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
          "barge_in_enabled": false,
          "barge_in_caller_says": "",
          "noise_enabled": false
        }}
      ]
    }}
  ]
}}

Produce these groups with these counts (≈20-22 scenarios total):
- baseline (2): clean happy path + one simple variation. No noise, no barge-in.
- confused_caller (4): each has at least one vague/ambiguous required field
  (e.g. "sometime next week", a mumbled number, a corrected date).
- impatient_caller (3): callers cut the agent off and demand speed; AT LEAST ONE
  must set barge_in_enabled=true with a realistic barge_in_caller_says.
- edge_cases (4): domain-specific hard cases (booking → double-book attempt,
  past date, fully-booked slot, holiday; lead-gen → budget refusal, wrong
  territory, already-a-customer; support → angry caller, out-of-scope request).
- audio_stress (3): ALL set noise_enabled=true; AT LEAST ONE sets
  barge_in_enabled=true. Caller speaks naturally over background noise.
- adversarial (4): caller gives WRONG info then corrects it, refuses a required
  field on first ask, or provides conflicting details mid-call.
{regression_instruction}
Rules:
- difficult_behaviors must have 3-5 specific, realistic items each
- ground_truth keys must match what pass_criteria check (e.g. "phone","date","time")
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite")
- persona_system_prompt ≤ 3 sentences (it becomes a TTS caller's system prompt)
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


_GROUNDED_BATTERY_USER_TEMPLATE = """\
AGENT DESCRIPTION:
{description}

CONFIRMED AGENT MODEL (this is the developer-reviewed source of truth — every
scenario you generate MUST target a specific part of it):

Goal: {goal}

Conversation graph nodes:
{nodes_block}

Tools the agent calls:
{tools_block}

Business rules the agent must obey:
{rules_block}

Detected risks to specifically test:
{risks_block}
{known_issues_block}
Generate a comprehensive stress-test BATTERY for this voice agent. Produce a
JSON object with this exact shape:
{{
  "category": "<snake_case label, e.g. appointment_booking>",
  "groups": [
    {{
      "group": "<one of: baseline | confused_caller | impatient_caller | edge_cases | audio_stress | adversarial | regression>",
      "scenarios": [
        {{
          "title": "<short title>",
          "node_id": "<the id of the graph node above this scenario primarily exercises>",
          "persona_system_prompt": "<2-3 sentences: who this caller is, their style>",
          "goal": "<what the caller wants, one sentence>",
          "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
          "ground_truth": {{"<key>": "<expected value>", ...}},
          "objective": "<what the agent must correctly do to pass, one sentence>",
          "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
          "barge_in_enabled": false,
          "barge_in_caller_says": "",
          "noise_enabled": false
        }}
      ]
    }}
  ]
}}

Produce these groups with these counts (≈20-22 scenarios total):
- baseline (2): clean happy path through the graph + one simple variation.
  No noise, no barge-in. node_id = the terminal success node.
- confused_caller (4): each has at least one vague/ambiguous required field
  from "required info" above (e.g. mumbled number, corrected date). node_id =
  the node where that field is collected.
- impatient_caller (3): callers cut the agent off and demand speed; AT LEAST ONE
  must set barge_in_enabled=true with a realistic barge_in_caller_says.
- edge_cases (4): target the RISKS and business rules listed above directly —
  each scenario should try to trigger one specific listed risk or violate one
  listed rule. node_id = the node closest to that risk/rule.
- audio_stress (3): ALL set noise_enabled=true; AT LEAST ONE sets
  barge_in_enabled=true.
- adversarial (4): caller gives WRONG info then corrects it, refuses a required
  field on first ask, or provides conflicting details mid-call.
{regression_instruction}
Rules:
- Every scenario's node_id MUST be one of the node ids listed above.
- Cover every node above with at least one scenario if the total count allows —
  do not cluster every scenario on the same node.
- difficult_behaviors must have 3-5 specific, realistic items each
- ground_truth keys must match what pass_criteria check (e.g. "phone","date","time")
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite")
- persona_system_prompt ≤ 3 sentences (it becomes a TTS caller's system prompt)
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


_FOCUSED_NODE_USER_TEMPLATE = """\
AGENT DESCRIPTION:
{description}

CONFIRMED AGENT MODEL (developer-reviewed source of truth):

Goal: {goal}

Conversation graph nodes:
{nodes_block}

Tools the agent calls:
{tools_block}

Business rules the agent must obey:
{rules_block}

Detected risks to specifically test:
{risks_block}

TARGET NODE (this is the ONLY node you should generate scenarios for):
{target_node_block}

Generate exactly {count} additional stress-test scenarios that ALL target the
TARGET NODE above. Vary difficulty and caller behavior across them (mix of
baseline, confused, impatient, edge-case, audio-stress, and adversarial
behaviors) so they cover the target node from many angles. Produce a JSON
object with this exact shape:
{{
  "category": "<snake_case label, e.g. appointment_booking>",
  "groups": [
    {{
      "group": "<one of: baseline | confused_caller | impatient_caller | edge_cases | audio_stress | adversarial>",
      "scenarios": [
        {{
          "title": "<short title>",
          "node_id": "{target_node_id}",
          "persona_system_prompt": "<2-3 sentences: who this caller is, their style>",
          "goal": "<what the caller wants, one sentence>",
          "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
          "ground_truth": {{"<key>": "<expected value>", ...}},
          "objective": "<what the agent must correctly do to pass, one sentence>",
          "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
          "barge_in_enabled": false,
          "barge_in_caller_says": "",
          "noise_enabled": false
        }}
      ]
    }}
  ]
}}

Rules:
- Every scenario's node_id MUST be exactly "{target_node_id}".
- Produce exactly {count} scenarios in total across the groups.
- difficult_behaviors must have 3-5 specific, realistic items each
- ground_truth keys must match what pass_criteria check (e.g. "phone","date","time")
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite")
- persona_system_prompt ≤ 3 sentences (it becomes a TTS caller's system prompt)
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


def _format_agent_model_blocks(agent_model: "AgentModel") -> dict[str, str]:
    """Render an AgentModel's fields into the bullet blocks the grounded prompt needs."""
    nodes_block = "\n".join(
        f"- {n.id}: {n.label}" + (f" — {n.objective}" if n.objective else "")
        for n in agent_model.nodes
    ) or "- (no graph nodes detected)"

    tools_block = "\n".join(
        f"- {t.name}" + (f" — {t.purpose}" if t.purpose else "")
        for t in agent_model.tools
    ) or "- (none)"

    rules_block = "\n".join(f"- {r.rule}" for r in agent_model.business_rules) or "- (none)"

    risks_block = "\n".join(
        f"- {r.description}" + (f" (near node: {r.node_id})" if r.node_id else "")
        for r in agent_model.risks
    ) or "- (none detected)"

    return {
        "goal": agent_model.goal or "(not specified)",
        "nodes_block": nodes_block,
        "tools_block": tools_block,
        "rules_block": rules_block,
        "risks_block": risks_block,
    }


_REGRESSION_INSTRUCTION = """\
- regression (2): target the user's KNOWN ISSUES described above. Build the exact
  caller behavior that would trigger each reported problem.
"""


_REGRESSION_TARGETED_USER_TEMPLATE = """\
FAILURE MODE TO TARGET:
{root_cause_description}

Generate scenarios that would trigger: {root_cause_description}

Produce a JSON object with this exact shape:
{{
  "category": "<snake_case label, e.g. api_timeout_regression>",
  "groups": [
    {{
      "group": "regression",
      "scenarios": [
        {{
          "title": "<short title>",
          "persona_system_prompt": "<2-3 sentences: who this caller is, their style>",
          "goal": "<what the caller wants, one sentence>",
          "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
          "ground_truth": {{"<key>": "<expected value>", ...}},
          "objective": "<what the agent must correctly do to pass, one sentence>",
          "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
          "barge_in_enabled": false,
          "barge_in_caller_says": "",
          "noise_enabled": false
        }}
      ]
    }}
  ]
}}

Rules:
- Produce exactly 5 scenarios, all in the "regression" group.
- Every scenario must be specifically engineered to reproduce the failure mode above —
  vary the exact trigger (timing, phrasing, data shape) across the 5 so each attacks
  it from a different angle.
- difficult_behaviors must directly reference the failure mode being targeted.
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite").
- persona_system_prompt <= 3 sentences (it becomes a TTS caller's system prompt).
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


_FAILURE_TRANSCRIPT_USER_TEMPLATE = """\
PRODUCTION CALL TRANSCRIPT THAT FAILED:
{transcript}

Analyze what went wrong in this real call, then generate regression scenarios
that would reproduce the same failure.

Produce a JSON object with this exact shape:
{{
  "category": "<snake_case label, e.g. calendar_lookup_regression>",
  "groups": [
    {{
      "group": "regression",
      "scenarios": [
        {{
          "title": "<short title>",
          "persona_system_prompt": "<2-3 sentences: who this caller is, their style>",
          "goal": "<what the caller wants, one sentence>",
          "difficult_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"],
          "ground_truth": {{"<key>": "<expected value>", ...}},
          "objective": "<what the agent must correctly do to pass, one sentence>",
          "pass_criteria": ["<criterion 1>", "<criterion 2>", "<criterion 3>"],
          "barge_in_enabled": false,
          "barge_in_caller_says": "",
          "noise_enabled": false
        }}
      ]
    }}
  ]
}}

Rules:
- Produce exactly 3 scenarios, all in the "regression" group.
- Base each scenario on the specific caller behavior, phrasing, and data from the
  transcript above — reproduce the same failure trigger, not a generic one.
- difficult_behaviors must directly reference what caused the failure in the transcript.
- pass_criteria must be objectively verifiable from a transcript (not "agent was polite").
- persona_system_prompt <= 3 sentences (it becomes a TTS caller's system prompt).
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
    return s.strip()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_variant(
    v: dict[str, Any],
    category: str,
    group: str = "",
    valid_node_ids: set[str] | None = None,
) -> Scenario:
    """Build a single Scenario from one variant dict (shared by both compile paths)."""
    title = v.get("title", f"{category} {group or 'variant'}".strip())
    scenario_id = f"{_slugify(title)}-{uuid.uuid4().hex[:6]}"

    barge_in_enabled = bool(v.get("barge_in_enabled", False))
    barge_in_says = v.get("barge_in_caller_says", "")
    noise_enabled = bool(v.get("noise_enabled", False))

    # Only trust node_id when it names a node that actually exists in the
    # confirmed model — otherwise a hallucinated id would silently break
    # graph coverage (a scenario "targeting" a node that isn't there).
    node_id = v.get("node_id", "") or ""
    if valid_node_ids is not None and node_id not in valid_node_ids:
        node_id = ""

    return Scenario(
        id=scenario_id,
        category=category,
        title=title,
        version=1,
        group=group,
        node_id=node_id,
        compiler_version=COMPILER_VERSION,
        persona=Persona(
            system_prompt=v["persona_system_prompt"],
            goal=v["goal"],
            difficult_behaviors=v["difficult_behaviors"],
            ground_truth=v.get("ground_truth", {}),
        ),
        noise=NoiseConfig(enabled=noise_enabled),
        barge_in=BargeInConfig(
            enabled=barge_in_enabled,
            caller_says=barge_in_says,
            on_agent_turn=3,
            trigger_after_agent_speaks_ms=800,
        ),
        scoring=ScoringConfig(
            objective=v["objective"],
            pass_criteria=v["pass_criteria"],
        ),
        limits=Limits(),
    )


def _build_scenarios(raw_json: str) -> list[Scenario]:
    data = json.loads(_strip_fences(raw_json))
    category = data.get("category", "custom")
    variants: list[dict[str, Any]] = data.get("variants", [])
    return [_build_variant(v, category) for v in variants]


def _build_battery(
    raw_json: str, valid_node_ids: set[str] | None = None
) -> list[Scenario]:
    data = json.loads(_strip_fences(raw_json))
    category = data.get("category", "custom")
    groups: list[dict[str, Any]] = data.get("groups", [])

    scenarios: list[Scenario] = []
    for grp in groups:
        group_name = grp.get("group", "custom")
        for v in grp.get("scenarios", []):
            scenarios.append(
                _build_variant(v, category, group=group_name, valid_node_ids=valid_node_ids)
            )
    return scenarios


# ---------------------------------------------------------------------------
# LLM implementation
# ---------------------------------------------------------------------------


class LLMScenarioCompiler:
    """
    Compile a description into 3 scenario variants using Claude or OpenAI.

    Uses the same model as the judge (LLM_JUDGE_MODEL) since this is a
    one-shot reasoning task — temperature 0, no streaming needed.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model

    async def compile(self, description: str) -> list[Scenario]:
        prompt = _USER_TEMPLATE.format(description=description.strip())

        if self._model.startswith("claude"):
            raw = await self._call_anthropic(prompt)
        else:
            raw = await self._call_openai(prompt)

        return _build_scenarios(raw)

    async def compile_battery(
        self,
        description: str,
        known_issues: str = "",
        agent_model: "AgentModel | None" = None,
        target_node_id: str = "",
        count: int = 0,
    ) -> list[Scenario]:
        known_issues = known_issues.strip()
        known_issues_block = (
            f"\nKNOWN ISSUES (reported by the user):\n{known_issues}\n"
            if known_issues
            else ""
        )
        regression_instruction = _REGRESSION_INSTRUCTION if known_issues else ""

        valid_node_ids: set[str] | None = None
        if agent_model is not None and agent_model.nodes:
            valid_node_ids = {n.id for n in agent_model.nodes}

        target_node = None
        if target_node_id and agent_model is not None:
            target_node = next(
                (n for n in agent_model.nodes if n.id == target_node_id), None
            )

        if target_node is not None:
            prompt = _FOCUSED_NODE_USER_TEMPLATE.format(
                description=description.strip(),
                target_node_id=target_node.id,
                target_node_block=f"- {target_node.id}: {target_node.label}"
                + (f" — {target_node.objective}" if target_node.objective else ""),
                count=count or 10,
                **_format_agent_model_blocks(agent_model),
            )
        elif agent_model is not None and agent_model.nodes:
            prompt = _GROUNDED_BATTERY_USER_TEMPLATE.format(
                description=description.strip(),
                known_issues_block=known_issues_block,
                regression_instruction=regression_instruction,
                **_format_agent_model_blocks(agent_model),
            )
        else:
            prompt = _BATTERY_USER_TEMPLATE.format(
                description=description.strip(),
                known_issues_block=known_issues_block,
                regression_instruction=regression_instruction,
            )

        print("\n" + "="*70)
        print("LLM SCENARIO COMPILER — SYSTEM PROMPT")
        print("="*70)
        print(_SYSTEM)
        print("="*70)
        print("LLM SCENARIO COMPILER — USER PROMPT")
        print("="*70)
        print(prompt)
        print("="*70 + "\n")

        if self._model.startswith("claude"):
            raw = await self._call_anthropic(prompt, max_tokens=8192)
        else:
            raw = await self._call_openai(prompt, max_tokens=8192)

        return _build_battery(raw, valid_node_ids=valid_node_ids)

    async def compile_regression(self, root_cause_description: str) -> list[Scenario]:
        prompt = _REGRESSION_TARGETED_USER_TEMPLATE.format(
            root_cause_description=root_cause_description.strip()
        )

        if self._model.startswith("claude"):
            raw = await self._call_anthropic(prompt, max_tokens=8192)
        else:
            raw = await self._call_openai(prompt, max_tokens=8192)

        return _build_battery(raw)

    async def compile_from_failure_text(self, transcript: str) -> list[Scenario]:
        prompt = _FAILURE_TRANSCRIPT_USER_TEMPLATE.format(transcript=transcript.strip())

        if self._model.startswith("claude"):
            raw = await self._call_anthropic(prompt, max_tokens=8192)
        else:
            raw = await self._call_openai(prompt, max_tokens=8192)

        return _build_battery(raw)

    async def _call_anthropic(self, prompt: str, max_tokens: int = 4096) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — pip install anthropic") from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self._model,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return message.content[0].text if message.content else ""

    async def _call_openai(self, prompt: str, max_tokens: int = 4096) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed — pip install openai") from e

        client = AsyncOpenAI(api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Stub (tests, no network)
# ---------------------------------------------------------------------------


class StubScenarioCompiler:
    """
    Returns a fixed list of 3 scenarios regardless of description.
    Used in tests and loopback mode to avoid API calls.
    """

    def __init__(
        self,
        scenarios: list[Scenario] | None = None,
        battery: list[Scenario] | None = None,
    ) -> None:
        self._scenarios = scenarios or _default_stub_scenarios()
        self._battery = battery

    async def compile(self, description: str) -> list[Scenario]:  # noqa: ARG002
        return list(self._scenarios)

    async def compile_battery(  # noqa: ARG002
        self,
        description: str,
        known_issues: str = "",
        agent_model: "AgentModel | None" = None,
        target_node_id: str = "",
        count: int = 0,
    ) -> list[Scenario]:
        if target_node_id:
            return [
                _stub_scenario(
                    f"node-{target_node_id}-{i}", "baseline", node_id=target_node_id
                )
                for i in range(count or 10)
            ]
        if self._battery is not None:
            return list(self._battery)
        return _default_stub_battery()

    async def compile_regression(self, root_cause_description: str) -> list[Scenario]:
        return [
            _stub_scenario(f"regression-{root_cause_description[:10]}-{i}", "regression")
            for i in range(5)
        ]

    async def compile_from_failure_text(self, transcript: str) -> list[Scenario]:
        return [
            _stub_scenario(f"failure-regression-{transcript[:10]}-{i}", "regression")
            for i in range(3)
        ]


def _stub_scenario(
    suffix: str,
    group: str,
    *,
    noise: bool = False,
    barge_in: bool = False,
    node_id: str = "",
) -> Scenario:
    return Scenario(
        id=f"stub-{suffix}-{uuid.uuid4().hex[:6]}",
        category="stub",
        title=f"Stub {group} {suffix}",
        version=1,
        group=group,
        node_id=node_id,
        compiler_version=COMPILER_VERSION,
        persona=Persona(
            system_prompt="You are a caller.",
            goal="Test the agent.",
            difficult_behaviors=["Be polite but slightly vague."],
        ),
        noise=NoiseConfig(enabled=noise),
        barge_in=BargeInConfig(enabled=barge_in, caller_says="Wait, hold on."),
        scoring=ScoringConfig(
            objective="Agent completes the task.",
            pass_criteria=["Agent responded correctly."],
        ),
    )


def _default_stub_scenarios() -> list[Scenario]:
    return [_stub_scenario("easy-000000", "baseline")]


def _default_stub_battery() -> list[Scenario]:
    return [
        _stub_scenario("baseline-1", "baseline"),
        _stub_scenario("baseline-2", "baseline"),
        _stub_scenario("confused-1", "confused_caller"),
        _stub_scenario("impatient-1", "impatient_caller", barge_in=True),
        _stub_scenario("edge-1", "edge_cases"),
        _stub_scenario("audio-1", "audio_stress", noise=True, barge_in=True),
        _stub_scenario("adversarial-1", "adversarial"),
    ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_compiler() -> LLMScenarioCompiler:
    """Instantiate the real compiler from config (reuses LLM_JUDGE_MODEL)."""
    from voicetest.config import Settings

    settings = Settings()
    chosen = settings.llm_judge_model or "claude-sonnet-4-6"

    if chosen.startswith("claude"):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for LLM scenario compiler")
        return LLMScenarioCompiler(api_key=settings.anthropic_api_key, model=chosen)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required for LLM scenario compiler")
    return LLMScenarioCompiler(api_key=settings.openai_api_key, model=chosen)
