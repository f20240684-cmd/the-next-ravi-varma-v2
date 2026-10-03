#!/usr/bin/env python3
"""Evaluate generated images against the held-out reference paintings.

Run evaluation (recommended; LoRA images vs. a no-LoRA baseline rendered
with the same prompt ids + seeds):
    python scripts/evaluate.py --input outputs/generated/lora --baseline outputs/generated/baseline \\
        --output-dir outputs/evaluations/lora_vs_base

Writes metrics.csv, summary.json, contact sheets and a qualitative_review.csv
template (1-5 human ratings for visual quality / anatomy / style / prompt).
See docs/EVALUATION.md for what each metric does and does not measure.

Single image:
    python scripts/evaluate.py --image outputs/generated/x.png --prompt "..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.evaluation.clip_score import clip_available
from ravi_varma.evaluation.report import evaluate_image
from ravi_varma.evaluation.run_eval import evaluate_run
from ravi_varma.utils.device import detect_device
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="outputs/generated", help="Directory of generated images (+ sidecar .json).")
    parser.add_argument("--baseline", default=None, help="Directory of base-model images with the same prompt ids/seeds.")
    parser.add_argument("--reference", default="data/reference", help="Held-out Ravi Varma paintings (never trained on).")
    parser.add_argument("--train-manifest", default="data/processed/manifest.jsonl", help="Training manifest, for the memorisation check.")
    parser.add_argument("--output-dir", default=None, help="Defaults to outputs/evaluations/<input dir name>.")
    parser.add_argument("--image", default=None, help="Evaluate a single image instead of a directory.")
    parser.add_argument("--prompt", default=None, help="Prompt for --image (defaults to its sidecar .json).")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = parser.parse_args()

    logger = setup_logging("evaluate")
    device = detect_device(args.device).device

    availability = clip_available(device=device)
    if not availability.available:
        logger.warning("CLIP unavailable (%s). Metrics will be reported as null, not fabricated.", availability.reason)

    if args.image:
        result = evaluate_image(args.image, args.reference, prompt=args.prompt, device=device)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    output_dir = Path(args.output_dir or Path("outputs/evaluations") / Path(args.input).name)
    try:
        run = evaluate_run(
            args.input, args.reference, output_dir,
            train_manifest=args.train_manifest, baseline_dir=args.baseline, device=device,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    print(json.dumps({"summary": run.summary, "baseline_comparison": run.baseline_comparison}, indent=2))
    print(f"\nFull report: {output_dir}/summary.json  |  human review sheet: {output_dir}/qualitative_review.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
