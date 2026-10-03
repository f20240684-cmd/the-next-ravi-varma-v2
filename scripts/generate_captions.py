#!/usr/bin/env python3
"""Generate structured, template-based captions for the dataset.

For every image accepted by dataset validation this writes a sidecar
`data/captions/<stem>.json` (all structured fields + every VQA question and
answer + the final flat caption) and stores the flat caption in the
`caption` field of data/metadata/metadata.jsonl.

Images that already have a VLM caption sidecar are skipped, so an
interrupted run can simply be restarted; pass --overwrite to redo them.

Usage:
    python scripts/generate_captions.py                 # accepted images, auto device
    python scripts/generate_captions.py --device cuda --overwrite
    python scripts/generate_captions.py --no-vlm        # offline pipeline test only (weak captions)
    python scripts/generate_captions.py --reflatten     # rebuild flat captions from saved sidecars
                                                        # (after editing titles or caption logic; no VLM needed)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import DatasetConfig
from ravi_varma.data.captions import StructuredCaption, clip_token_counter, get_captioner
from ravi_varma.data.metadata import load_metadata, save_metadata
from ravi_varma.data.validation import DatasetValidator
from ravi_varma.utils.device import detect_device
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--vlm-model", default="Salesforce/blip-image-captioning-large")
    parser.add_argument("--vqa-model", default="Salesforce/blip-vqa-base", help="Set to 'none' to skip the template questions.")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--all", action="store_true", help="Caption every raw image, including ones validation excluded.")
    parser.add_argument("--overwrite", action="store_true", help="Re-caption images that already have a VLM caption.")
    parser.add_argument("--no-vlm", action="store_true", help="Use the offline heuristic captioner (pipeline testing only; not for training).")
    parser.add_argument("--reflatten", action="store_true", help="Re-assemble flat captions from existing VLM sidecars + current metadata titles.")
    parser.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Model whose CLIP tokenizer is used to fit captions into 77 tokens.")
    args = parser.parse_args()

    logger = setup_logging("generate_captions")
    config = DatasetConfig.load(args.config)
    raw_dir = Path(config.raw_dir)
    captions_dir = Path(config.captions_dir)
    captions_dir.mkdir(parents=True, exist_ok=True)

    report = DatasetValidator(config).validate()
    if args.all:
        images = sorted([*report.accepted, *report.excluded])
    else:
        images = report.accepted
        logger.info("Captioning %d accepted images (%d excluded by validation; use --all to include them).", len(images), len(report.excluded))
    if not images:
        logger.error("No images to caption in %s. Run scripts/validate_dataset.py for details.", raw_dir)
        return 1

    metadata = {r["image"]: r for r in load_metadata(config.metadata_file)}

    if args.reflatten:
        count_tokens = clip_token_counter(args.base_model)
        updated = 0
        for rel in images:
            sidecar = captions_dir / f"{Path(rel).stem}.json"
            if not sidecar.exists():
                logger.warning("No caption sidecar for %s; run without --reflatten first.", rel)
                continue
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data.get("source") != "vlm":
                continue
            record = metadata.setdefault(rel, {"image": rel, "artist": config.style.artist_name})
            structured = StructuredCaption.reassemble(data, title=record.get("title"))
            flat = structured.as_flat_caption(token_counter=count_tokens)
            sidecar.write_text(
                json.dumps({"image": rel, **structured.to_dict(), "flat_caption": flat}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            record["caption"] = flat
            updated += 1
        save_metadata(config.metadata_file, metadata.values())
        logger.info("Re-flattened %d captions -> %s", updated, config.metadata_file)
        return 0

    todo = []
    for rel in images:
        sidecar = captions_dir / f"{Path(rel).stem}.json"
        if not args.overwrite and sidecar.exists():
            existing = json.loads(sidecar.read_text(encoding="utf-8"))
            if existing.get("source") == "vlm" and existing.get("flat_caption"):
                metadata.setdefault(rel, {"image": rel, "artist": config.style.artist_name})["caption"] = existing["flat_caption"]
                continue
        todo.append(rel)
    logger.info("%d images need captions (%d already done).", len(todo), len(images) - len(todo))

    if todo:
        device = detect_device(args.device).device
        try:
            captioner = get_captioner(
                prefer_vlm=not args.no_vlm,
                model_id=args.vlm_model,
                vqa_model_id=None if args.vqa_model.lower() == "none" else args.vqa_model,
                device=device,
            )
        except RuntimeError as e:
            logger.error("%s", e)
            return 2
        count_tokens = clip_token_counter(args.base_model)
        logger.info("Using %s on %s", type(captioner).__name__, device)

        start = time.time()
        for i, rel in enumerate(todo, 1):
            record = metadata.setdefault(rel, {"image": rel, "artist": config.style.artist_name})
            try:
                structured = captioner.caption(raw_dir / rel, title=record.get("title"))
            except ValueError as e:
                logger.warning("Skipping %s: %s", rel, e)
                continue
            flat = structured.as_flat_caption(token_counter=count_tokens)
            sidecar = captions_dir / f"{Path(rel).stem}.json"
            sidecar.write_text(
                json.dumps({"image": rel, **structured.to_dict(), "flat_caption": flat}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Heuristic captions are written to the sidecar for inspection but
            # never promoted into the training metadata.
            if structured.source == "vlm":
                record["caption"] = flat
            logger.info("[%d/%d] %s -> %s", i, len(todo), rel, flat)
            if i % 20 == 0:
                save_metadata(config.metadata_file, metadata.values())  # checkpoint progress
        logger.info("Captioned %d images in %.0fs", len(todo), time.time() - start)

    save_metadata(config.metadata_file, metadata.values())
    logger.info("Updated captions in %s", config.metadata_file)
    logger.info("Next: python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
