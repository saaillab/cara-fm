"""CLI entrypoint for inspecting CARA-FM index definitions and benchmark scorecards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .indices import evaluate_system_card
from .schema import validate_scorecard_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="CARA-FM Nine-Index Evaluation CLI")
    parser.add_argument("--validate-json", type=Path, help="Validate a CARA-FM scorecard or benchmark JSON file")
    parser.add_argument("--demo", action="store_true", help="Print a sample edge-model evaluation scorecard")
    args = parser.parse_args()

    if args.demo:
        card = evaluate_system_card(
            model_id="MobileNetV3-Small-INT8-Ref",
            task_domain="east_african_crop_grading",
            balanced_accuracy=88.5,
            model_size_mb=9.8,
            peak_ram_gb=0.35,
            cpu_latency_ms=28.0,
            offline_capable=True,
            requires_gpu=False,
            local_language_scores=[0.84, 0.78],
            high_resource_lang_score=0.92,
            local_taxonomy_coverage=1.0,
            swahili_glossary_parity=0.88,
            offline_ui_ready=1.0,
            iks_compatibility=0.75,
        )
        data = card.to_dict()
        errs = validate_scorecard_dict(data)
        assert not errs, errs
        print(json.dumps(data, indent=2))
        return

    if args.validate_json:
        raw = json.loads(args.validate_json.read_text(encoding="utf-8"))
        items = raw.get("models", [raw]) if isinstance(raw, dict) else raw
        all_errs = []
        for idx, item in enumerate(items):
            errs = validate_scorecard_dict(item)
            if errs:
                all_errs.extend(f"[{idx}] {e}" for e in errs)
        if all_errs:
            for e in all_errs:
                print("ERROR:", e)
            raise SystemExit(1)
        print(f"Validated {len(items)} CARA-FM scorecard(s) with 0 schema errors.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
