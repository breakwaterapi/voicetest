"""
Scenario YAML loader.

load_scenario(path)  — parse + validate one file, fail loudly on bad input.
load_all(dir)        — load every *.yaml in a directory (non-recursive).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from voicetest.scenarios.models import Scenario

# Directory containing the package-shipped builtin scenarios
BUILTIN_DIR = Path(__file__).parent / "builtin"


def load_scenario(path: Path | str) -> Scenario:
    """
    Parse and validate a scenario YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    pydantic.ValidationError
        If required fields are missing, types are wrong, difficult_behaviors is
        empty, or unknown keys are present (extra='forbid').
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file must be a YAML mapping, got {type(raw).__name__}: {path}")
    return Scenario.model_validate(raw)


def load_all(directory: Path | str | None = None) -> list[Scenario]:
    """
    Load all *.yaml files in *directory* (defaults to the builtin scenarios dir).

    Returns scenarios in filename-alphabetical order for determinism.
    Raises on any file that fails validation — no silent skips.
    """
    directory = Path(directory) if directory is not None else BUILTIN_DIR
    paths = sorted(directory.glob("*.yaml"))
    return [load_scenario(p) for p in paths]
