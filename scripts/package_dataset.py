#!/usr/bin/env python3
"""Package the prepared dataset for training on another machine (Colab).

data/raw/ (~400 MB of museum scans) is not in git. Training only needs the
prepared, resized images + manifest, so this bundles:

    data/processed/   resized training images + manifest.jsonl
    data/reference/   held-out evaluation paintings + reference.jsonl
    data/captions/    per-image structured captions (audit trail)
    data/metadata/    metadata.jsonl + validation_report.json

into dist/ravi_varma_dataset.zip (~30 MB). Upload it to Colab / Google
Drive and unzip it in the project root (see notebooks/03_lora_training.ipynb).

Usage:
    python scripts/package_dataset.py [--output dist/ravi_varma_dataset.zip]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--output", default="dist/ravi_varma_dataset.zip")
    args = parser.parse_args()

    logger = setup_logging("package_dataset")
    config = DatasetConfig.load(args.config)
    manifest = Path(config.manifest_file)
    if not manifest.exists():
        logger.error("No manifest at %s; run scripts/prepare_dataset.py first.", manifest)
        return 1

    dirs = [Path(config.processed_dir), Path(config.reference_dir), Path(config.captions_dir), Path(config.metadata_dir)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in dirs:
            for p in sorted(d.rglob("*")):
                if p.is_file() and not p.name.startswith("."):
                    zf.write(p, p.as_posix())
                    count += 1
    logger.info("Wrote %d files (%.1f MB) -> %s", count, out.stat().st_size / 1e6, out)
    logger.info("On Colab: upload it, then `!unzip -o ravi_varma_dataset.zip` in the project root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
