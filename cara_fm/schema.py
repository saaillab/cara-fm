"""Lightweight built-in validator matching schemas/cara_fm_evaluation_schema.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "cara_fm_evaluation_schema.json"


def load_schema() -> Dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_scorecard_dict(payload: Dict[str, object]) -> List[str]:
    """Validate a scorecard dict against required CARA-FM schema fields and [0, 1] bounds."""
    errors: List[str] = []
    required_top = [
        "framework_version",
        "license",
        "model_id",
        "hardware_profile",
        "technical_indices",
        "agricultural_indices",
        "cultural_indices",
        "composite_readiness_score",
    ]
    for k in required_top:
        if k not in payload:
            errors.append(f"Missing top-level field: {k}")

    if payload.get("license") != "CC-BY-4.0":
        errors.append(f"Invalid license: {payload.get('license')} (expected CC-BY-4.0)")

    tech = payload.get("technical_indices", {})
    if isinstance(tech, dict):
        for idx in ("CEI", "DFS", "MPP"):
            val = tech.get(idx)
            if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
                errors.append(f"Invalid technical index {idx}: {val}")

    cult = payload.get("cultural_indices", {})
    if isinstance(cult, dict):
        cai = cult.get("CAI")
        if not isinstance(cai, (int, float)) or not (0.0 <= float(cai) <= 1.0):
            errors.append(f"Invalid CAI: {cai}")
        for opt in ("CAS", "KPV"):
            v = cult.get(opt)
            if v is not None and (not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0)):
                errors.append(f"Invalid cultural trial index {opt}: {v}")

    ag = payload.get("agricultural_indices", {})
    if isinstance(ag, dict):
        for opt in ("YIP", "IOE", "IAE"):
            v = ag.get(opt)
            if v is not None and (not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0)):
                errors.append(f"Invalid agricultural trial index {opt}: {v}")

    comp = payload.get("composite_readiness_score")
    if not isinstance(comp, (int, float)) or not (0.0 <= float(comp) <= 1.0):
        errors.append(f"Invalid composite_readiness_score: {comp}")

    return errors
