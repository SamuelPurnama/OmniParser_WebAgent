#!/usr/bin/env python3
"""Minimal CLI entry point for running the Gemini Computer-Use agent."""
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.services.models import InstructionRequest
from src.pipeline.services.pipeline_runner import pipeline_runner


def _load_instruction(config_path: Path, overrides: Dict[str, Any]) -> InstructionRequest:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.update({k: v for k, v in overrides.items() if v is not None})

    # Normalise comma-separated fields if provided as strings
    if isinstance(data.get("devices"), str):
        data["devices"] = [d.strip() for d in data["devices"].split(",") if d.strip()]
    if isinstance(data.get("browsers"), str):
        data["browsers"] = [b.strip() for b in data["browsers"].split(",") if b.strip()]

    return InstructionRequest(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemini UI agent pipeline without Django.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON file containing the InstructionRequest payload.",
    )
    parser.add_argument(
        "--episode-name",
        type=str,
        help="Optional name for the saved episode directory (defaults to task name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/pipeline/data/results/manual_run"),
        help="Directory where screenshots + metadata will be stored.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        help="Optional comma-separated override for devices (e.g. 'desktop,mobile').",
    )
    parser.add_argument(
        "--browsers",
        type=str,
        help="Optional comma-separated override for browsers (e.g. 'chrome,firefox').",
    )
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.devices:
        overrides["devices"] = [d.strip() for d in args.devices.split(",") if d.strip()]
    if args.browsers:
        overrides["browsers"] = [b.strip() for b in args.browsers.split(",") if b.strip()]

    instruction = _load_instruction(args.config, overrides)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    response = pipeline_runner.run_pipeline(
        instruction=instruction,
        episode_name=args.episode_name or instruction.task.lower().replace(" ", "_"),
        base_episode_dir=str(args.output_dir),
    )

    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()

