"""Dataset preparation: turns validated raw images + captions into a
processed, resized image directory and a JSONL manifest suitable both for
Hugging Face `datasets.Dataset.from_json` and for the lightweight
`RaviVarmaImageCaptionDataset` used directly by the LoRA trainer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator, Optional

from ravi_varma.config import DatasetConfig
from ravi_varma.data.metadata import load_metadata
from ravi_varma.utils.image import (
    aspect_buckets,
    nearest_bucket,
    resize_for_training,
    resize_to_bucket,
    safe_open_image,
    trim_uniform_border,
)
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ManifestRecord:
    image: str
    text: str
    artist: str = "Raja Ravi Varma"
    source: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    source_image: Optional[str] = None  # raw file this sample was made from (relative to raw_dir)
    width: Optional[int] = None
    height: Optional[int] = None
    license: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _load_metadata_records(metadata_file: Path) -> list[dict]:
    return load_metadata(metadata_file)


def _load_caption_for_image(rel_image_path: str, captions_dir: Path, metadata_record: Optional[dict]) -> Optional[str]:
    """Caption resolution order: metadata JSONL `caption` field takes
    priority (generate_captions.py writes it there); otherwise fall back to
    a sidecar .txt file, or a .json sidecar produced by a real VLM. Sidecars
    from the heuristic captioner are ignored -- they carry no content
    information and must not be trained on."""
    if metadata_record and metadata_record.get("caption"):
        return str(metadata_record["caption"]).strip()

    stem = Path(rel_image_path).stem
    txt_path = captions_dir / f"{stem}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip() or None

    json_path = captions_dir / f"{stem}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("source", "vlm") == "vlm":
            return data.get("flat_caption") or data.get("raw_caption")
    return None


def _clear_processed_dir(processed_dir: Path) -> None:
    """Remove images from a previous preparation run so renamed/excluded
    samples can't linger next to the new manifest."""
    for p in processed_dir.glob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            p.unlink()


def prepare_dataset(config: DatasetConfig, accepted: Optional[list[str]] = None) -> list[ManifestRecord]:
    """Turn accepted raw images + captions into training samples.

    `accepted` is the list of raw image paths (relative to raw_dir) that
    passed validation; `scripts/prepare_dataset.py` passes the validator's
    decision. If omitted, every readable image in raw_dir is considered
    (useful for tests / custom pipelines).

    Each image is optionally border-trimmed, then resized either to the
    closest aspect-ratio bucket (default) or to a square, saved to
    `processed_dir/<raw stem>.jpg`, and listed in the JSONL manifest with
    its trigger-prefixed caption and provenance.

    Images listed in `curation.holdout` are processed identically but
    written to `reference_dir` (with `reference.jsonl`) instead of the
    training manifest -- they are the held-out evaluation set.
    """
    raw_dir = Path(config.raw_dir)
    processed_dir = Path(config.processed_dir)
    captions_dir = Path(config.captions_dir)
    manifest_file = Path(config.manifest_file)
    pre = config.preprocessing

    reference_dir = Path(config.reference_dir)
    holdout = set(config.curation.holdout)

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    _clear_processed_dir(processed_dir)
    if holdout:
        reference_dir.mkdir(parents=True, exist_ok=True)
        _clear_processed_dir(reference_dir)

    metadata_by_image = {str(Path(r["image"])): r for r in _load_metadata_records(Path(config.metadata_file))}

    if accepted is None:
        allowed = {ext.lower() for ext in config.validation.allowed_formats}
        accepted = sorted(
            str(p.relative_to(raw_dir)) for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed
        )
    buckets = aspect_buckets(pre.target_resolution) if pre.aspect_bucketing else None

    manifest: list[ManifestRecord] = []
    reference: list[ManifestRecord] = []
    skipped = 0
    for rel in accepted:
        img_path = raw_dir / rel
        record = metadata_by_image.get(rel) or {}

        caption = _load_caption_for_image(rel, captions_dir, record)
        if not caption:
            skipped += 1
            logger.warning("Skipping %s: no caption found (run scripts/generate_captions.py).", rel)
            continue

        # Decode large JPEG scans at reduced size: we only need ~2x the
        # training resolution for a high-quality Lanczos downscale.
        img = safe_open_image(img_path, max_side=pre.target_resolution * 2)
        if img is None:
            skipped += 1
            logger.warning("Skipping %s: image is unreadable/corrupt.", rel)
            continue
        if pre.trim_borders:
            img = trim_uniform_border(img)

        if buckets:
            processed = resize_to_bucket(img, nearest_bucket(*img.size, buckets), pre.interpolation)
        else:
            processed = resize_for_training(img, pre.target_resolution, pre.resize_mode, pre.interpolation)

        out_path = (reference_dir if rel in holdout else processed_dir) / f"{Path(rel).stem}.jpg"
        processed.save(out_path, format="JPEG", quality=95)

        note = config.curation.caption_notes.get(rel)
        if note:
            caption = f"{caption}, {note}"
        prefixed_caption = config.style.caption_prefix_template.format(
            trigger_token=config.style.trigger_token, caption=caption
        )
        (reference if rel in holdout else manifest).append(
            ManifestRecord(
                image=str(out_path),
                text=prefixed_caption,
                artist=record.get("artist") or config.style.artist_name,
                source=record.get("source"),
                year=record.get("year"),
                title=record.get("title"),
                source_image=rel,
                width=processed.width,
                height=processed.height,
                license=record.get("license"),
            )
        )

    with manifest_file.open("w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    if holdout:
        with (reference_dir / "reference.jsonl").open("w", encoding="utf-8") as f:
            for rec in reference:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        missing = holdout - {rec.source_image for rec in reference}
        if missing:
            logger.warning("Hold-out images not prepared (not accepted or uncaptioned): %s", sorted(missing))
        logger.info("Held out %d images for evaluation -> %s", len(reference), reference_dir)

    sizes: dict[str, int] = {}
    for rec in manifest:
        key = f"{rec.width}x{rec.height}"
        sizes[key] = sizes.get(key, 0) + 1
    logger.info(
        "Prepared %d training samples (%d skipped) -> %s | resolutions: %s",
        len(manifest), skipped, manifest_file, dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
    )
    return manifest


def load_manifest(manifest_file: "str | Path") -> list[ManifestRecord]:
    manifest_file = Path(manifest_file)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_file}. Run scripts/prepare_dataset.py first."
        )
    records = []
    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = ManifestRecord.from_dict(json.loads(line))
                # Paths are written relative to the project root; also accept
                # a manifest that was moved together with its images.
                if not Path(rec.image).exists() and (manifest_file.parent / Path(rec.image).name).exists():
                    rec.image = str(manifest_file.parent / Path(rec.image).name)
                records.append(rec)
    return records


def iter_hf_dataset_dicts(manifest: list[ManifestRecord]) -> Iterator[dict]:
    """Yield plain dicts in the shape `datasets.Dataset.from_list` (or
    `Dataset.from_generator`) expects: {"image": <path>, "text": <caption>}.
    Kept as a thin adapter so `datasets` remains an optional dependency for
    everything except the actual training script."""
    for rec in manifest:
        yield {"image": rec.image, "text": rec.text}


class RaviVarmaImageCaptionDataset:
    """A minimal torch Dataset over the manifest, used by the LoRA trainer.

    With `keep_aspect=True` (the default, matching aspect-bucketed
    preparation) each processed image is used at its prepared size, e.g.
    448x640, so batches must group equal sizes -- see `BucketBatchSampler`.
    With `keep_aspect=False` every image is resized + cropped to a
    `resolution` square (the original behaviour).

    Implemented without inheriting from `torch.utils.data.Dataset` at import
    time so this module stays importable without torch; the class only
    requires torch inside `__getitem__`.
    """

    def __init__(
        self,
        manifest: list[ManifestRecord],
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        keep_aspect: bool = True,
    ):
        self.manifest = manifest
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.keep_aspect = keep_aspect

    def __len__(self) -> int:
        return len(self.manifest)

    def size_of(self, idx: int) -> tuple[int, int]:
        """(width, height) the sample will have, without decoding it."""
        rec = self.manifest[idx]
        if self.keep_aspect and rec.width and rec.height:
            return rec.width // 8 * 8, rec.height // 8 * 8
        return self.resolution, self.resolution

    def __getitem__(self, idx: int) -> dict:
        import random

        from torchvision import transforms

        rec = self.manifest[idx]
        img = safe_open_image(rec.image)
        if img is None:
            raise RuntimeError(f"Could not read processed image {rec.image}; re-run prepare_dataset.py.")

        if self.keep_aspect:
            w, h = self.size_of(idx)
            tfs = [] if img.size == (w, h) else [transforms.Resize((h, w), interpolation=transforms.InterpolationMode.LANCZOS)]
        else:
            tfs = [transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.LANCZOS)]
            tfs.append(
                transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution)
            )
        if self.random_flip and random.random() < 0.5:
            tfs.append(transforms.RandomHorizontalFlip(p=1.0))
        tfs += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]

        pixel_values = transforms.Compose(tfs)(img)
        return {"pixel_values": pixel_values, "text": rec.text}


class BucketBatchSampler:
    """Yields batches of indices whose samples share one resolution, so
    aspect-bucketed images can be stacked. Shuffles within and across
    buckets every epoch, seeded for reproducibility. With batch_size=1
    (the project default) this is equivalent to a plain shuffle."""

    def __init__(self, dataset: RaviVarmaImageCaptionDataset, batch_size: int, seed: int = 0, drop_last: bool = False):
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for idx in range(len(dataset)):
            self.buckets.setdefault(dataset.size_of(idx), []).append(idx)

    def _batches(self) -> list[list[int]]:
        import random

        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.buckets.values():
            indices = indices[:]
            rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())
