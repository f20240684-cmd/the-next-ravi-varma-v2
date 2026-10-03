#!/usr/bin/env python3
"""Build the training set: validated + captioned raw images -> resized
(aspect-bucketed) JPEGs in data/processed/ and data/processed/manifest.jsonl.

Runs dataset validation first and prepares exactly the images it accepts,
so curation exclusions and duplicate removal always apply.

Usage:
    python scripts/prepare_dataset.py [--config configs/dataset.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.data.dataset import prepare_dataset
from ravi_varma.data.validation import DatasetValidator
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--skip-validation", action="store_true", help="Prepare every readable raw image (ignores curation and duplicate removal).")
    args = parser.parse_args()

    logger = setup_logging("prepare_dataset")
    config = DatasetConfig.load(args.config)

    accepted = None
    if not args.skip_validation:
        report = DatasetValidator(config).validate()
        if not report.is_valid:
            logger.error("Dataset has 0 accepted images; refusing to prepare. Run scripts/validate_dataset.py for details.")
            return 1
        accepted = report.accepted
        logger.info(
            "Validation accepted %d/%d images (excluded: %s).",
            len(accepted), report.total_images_found, report.exclusion_counts(),
        )

    manifest = prepare_dataset(config, accepted=accepted)
    if not manifest:
        logger.error("No samples were prepared (missing captions?). Run scripts/generate_captions.py first.")
        return 1
    if accepted is not None:
        expected = len([a for a in accepted if a not in set(config.curation.holdout)])
        if len(manifest) < expected:
            logger.warning("%d accepted training images were skipped (see warnings above).", expected - len(manifest))

    logger.info("Prepared %d samples -> %s", len(manifest), config.manifest_file)
    logger.info("Next: python scripts/train_lora.py --config configs/lora_sd15.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
