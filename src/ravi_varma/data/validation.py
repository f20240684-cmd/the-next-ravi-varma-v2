"""Dataset validation and curation.

Used by `scripts/validate_dataset.py`, and re-run by `generate_captions.py`
and `prepare_dataset.py` so all three agree on exactly which images are in
the training set. Checks a raw image directory (plus the metadata JSONL) for
the problems most likely to break or bias LoRA training:

  * unsupported formats, corrupt/truncated files, images below a minimum
    resolution (rejected) -- including local copies that were *upscaled*
    from a smaller original recorded in metadata -- and very large files
    (flagged; downscaled later)
  * metadata that references missing files, duplicate metadata rows, rows
    without provenance
  * manual curation exclusions from `configs/dataset.yaml`
  * exact duplicates (identical bytes), curated duplicate groups (same
    artwork, different scan/crop) and automatic near-duplicates (difference
    hash) -- for each duplicate set only the highest-resolution copy is kept
  * missing captions (a warning here, because captioning runs *after*
    validation; `prepare_dataset` refuses uncaptioned images)

The outcome is an explicit `accepted` list and an `excluded` {image: reason}
map, saved in the JSON report so every exclusion is auditable.

Only depends on Pillow/numpy (via `ravi_varma.utils.image`) -- no torch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ravi_varma.config import DatasetConfig
from ravi_varma.utils.image import difference_hash, file_sha256, hamming_distance, safe_open_image
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_METADATA_FIELDS = {"image"}
OPTIONAL_METADATA_FIELDS = {"caption", "artist", "source", "year", "title"}


@dataclass
class DatasetIssue:
    severity: str  # "error" | "warning"
    category: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category, "path": self.path, "message": self.message}


@dataclass
class ValidationReport:
    total_images_found: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    duplicate_sets: list[dict] = field(default_factory=list)
    image_info: dict[str, dict] = field(default_factory=dict)

    def add(self, severity: str, category: str, path: str, message: str) -> None:
        self.issues.append(DatasetIssue(severity, category, path, message))

    def exclude(self, image: str, category: str, reason: str, severity: str = "warning") -> None:
        if image not in self.excluded:
            self.excluded[image] = f"{category}: {reason}"
        self.add(severity, category, image, reason)

    @property
    def valid_samples(self) -> int:
        return len(self.accepted)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_valid(self) -> bool:
        """Dataset is trainable (not necessarily perfect) if at least one
        image survived validation."""
        return self.valid_samples > 0

    def exclusion_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for reason in self.excluded.values():
            category = reason.split(":", 1)[0]
            counts[category] = counts.get(category, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {
            "total_images_found": self.total_images_found,
            "valid_samples": self.valid_samples,
            "excluded_count": len(self.excluded),
            "exclusions_by_category": self.exclusion_counts(),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "is_valid": self.is_valid,
            "accepted": self.accepted,
            "excluded": self.excluded,
            "duplicate_sets": self.duplicate_sets,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for issue in self.issues:
            key = f"{issue.severity}:{issue.category}"
            counts[key] = counts.get(key, 0) + 1
        lines = [
            "=== Dataset Validation Report ===",
            f"Images found:     {self.total_images_found}",
            f"Accepted:         {self.valid_samples}",
            f"Excluded:         {len(self.excluded)}  {self.exclusion_counts()}",
            f"Errors:           {self.error_count}",
            f"Warnings:         {self.warning_count}",
            "",
            "Issues by category:",
        ]
        lines += [f"  {k:40s} {v}" for k, v in sorted(counts.items())]
        if self.excluded:
            lines += ["", "Excluded images:"]
            lines += [f"  {img:12s} {reason}" for img, reason in sorted(self.excluded.items())]
        return "\n".join(lines)


def _load_metadata(metadata_file: Path) -> list[dict]:
    if not metadata_file.exists():
        return []
    records = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {metadata_file}:{lineno}: {e}") from e
            if not REQUIRED_METADATA_FIELDS.issubset(record.keys()):
                raise ValueError(
                    f"Metadata record at {metadata_file}:{lineno} missing required field(s): "
                    f"{REQUIRED_METADATA_FIELDS - record.keys()}"
                )
            records.append(record)
    return records


class DatasetValidator:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        raw_dir = Path(self.config.raw_dir)
        metadata_file = Path(self.config.metadata_file)
        cfg = self.config.validation

        if not raw_dir.exists():
            report.add("error", "missing_directory", str(raw_dir), "Raw image directory does not exist.")
            return report

        # 1. Discover candidate image files on disk.
        all_files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
        allowed = {ext.lower() for ext in cfg.allowed_formats}
        image_candidates = [p for p in all_files if p.suffix.lower() in allowed]
        for p in all_files:
            if p.suffix.lower() not in allowed:
                report.add("warning", "unsupported_format", str(p.relative_to(raw_dir)), f"Unsupported file extension '{p.suffix}'; ignored.")

        report.total_images_found = len(image_candidates)
        if report.total_images_found == 0:
            report.add("error", "empty_dataset", str(raw_dir), "No images with an allowed extension were found. Add images to data/raw/.")
            return report

        # 2. Metadata consistency.
        try:
            metadata_records = _load_metadata(metadata_file)
        except ValueError as e:
            report.add("error", "invalid_metadata", str(metadata_file), str(e))
            metadata_records = []

        metadata_by_image: dict[str, dict] = {}
        for record in metadata_records:
            key = str(Path(record["image"]))
            if key in metadata_by_image:
                report.add("warning", "duplicate_metadata", key, "Image appears more than once in metadata; the last row wins.")
            metadata_by_image[key] = record
            if not (raw_dir / key).exists():
                report.add("error", "missing_image", key, "Metadata references an image that does not exist on disk.")
            elif not record.get("source"):
                report.add("warning", "missing_provenance", key, "No `source` (provenance/license) recorded for this image.")

        if not metadata_records and cfg.require_caption:
            report.add(
                "warning",
                "no_metadata_file",
                str(metadata_file),
                "No metadata file found; captions/provenance cannot be validated. "
                "Run scripts/generate_captions.py or provide data/metadata/metadata.jsonl.",
            )

        # 3. Per-image checks: readability, resolution, size, hashes.
        candidates: list[str] = []
        sha_groups: dict[str, list[str]] = {}
        dhashes: dict[str, str] = {}
        for img_path in image_candidates:
            rel = str(img_path.relative_to(raw_dir))

            size_mb = img_path.stat().st_size / (1024 * 1024)
            try:
                with Image.open(img_path) as header:
                    width, height = header.size
            except Exception:
                report.exclude(rel, "corrupt_image", "File is not a readable image.", severity="error")
                continue
            report.image_info[rel] = {"width": width, "height": height, "size_mb": round(size_mb, 2)}

            if min(width, height) < cfg.min_resolution:
                report.exclude(rel, "low_resolution", f"Shortest side {min(width, height)}px is below the minimum {cfg.min_resolution}px.", severity="error")
                continue

            meta = metadata_by_image.get(rel) or {}
            src_w, src_h = meta.get("source_width"), meta.get("source_height")
            if src_w and src_h and src_w * src_h < width * height:
                report.image_info[rel].update(source_width=src_w, source_height=src_h)
                if min(src_w, src_h) < cfg.min_resolution:
                    report.exclude(
                        rel,
                        "upscaled_low_resolution",
                        f"Local file is {width}x{height} but the original is only {src_w}x{src_h}; "
                        f"the upscale adds no detail and the original is below {cfg.min_resolution}px.",
                    )
                    continue
                report.add("warning", "upscaled_copy", rel, f"Local file ({width}x{height}) is an upscale of the {src_w}x{src_h} original.")

            # Decoding at reduced size still reads the whole stream, so
            # truncated files are caught without decoding 100MP at full size.
            img = safe_open_image(img_path, max_side=256)
            if img is None:
                report.exclude(rel, "corrupt_image", "Image is corrupt, truncated, or unreadable.", severity="error")
                continue

            if size_mb > cfg.max_file_size_mb:
                report.add("warning", "large_file", rel, f"File is {size_mb:.1f}MB (> {cfg.max_file_size_mb}MB); accepted, will be downscaled during preparation.")

            sha_groups.setdefault(file_sha256(img_path), []).append(rel)
            if cfg.detect_duplicates:
                h = difference_hash(img)
                if h is not None:
                    dhashes[rel] = h

            candidates.append(rel)

        # 4. Manual curation exclusions.
        known = set(candidates)
        for image, reason in self.config.curation.exclude.items():
            if image in known:
                report.exclude(image, "curated_exclusion", reason)
            elif image not in report.excluded:
                report.add("warning", "stale_curation_entry", image, "Listed in curation.exclude but not found among the raw images.")

        # 5. Duplicates: exact bytes, curated groups, automatic near-duplicates.
        duplicate_sets: list[tuple[str, list[str]]] = []
        for paths in sha_groups.values():
            if len(paths) > 1:
                duplicate_sets.append(("exact_duplicate", paths))
        for group in self.config.curation.duplicate_groups:
            present = [p for p in group if p in known]
            if len(present) > 1:
                duplicate_sets.append(("curated_duplicate", present))
        if cfg.detect_duplicates:
            items = sorted(dhashes.items())
            for i, (a, ha) in enumerate(items):
                for b, hb in items[i + 1:]:
                    distance = hamming_distance(ha, hb)
                    if distance <= cfg.near_duplicate_threshold:
                        duplicate_sets.append(("near_duplicate", [a, b]))

        for category, paths in self._merge_duplicate_sets(duplicate_sets):
            alive = [p for p in paths if p not in report.excluded]
            if len(alive) < 2:
                continue
            keep = max(alive, key=lambda p: report.image_info[p]["width"] * report.image_info[p]["height"])
            dropped = [p for p in alive if p != keep]
            report.duplicate_sets.append({"kind": category, "kept": keep, "excluded": dropped})
            for p in dropped:
                report.exclude(p, category, f"Same artwork as {keep} (kept the higher-resolution copy).")

        report.accepted = [p for p in candidates if p not in report.excluded]
        if cfg.require_caption and metadata_records:
            for rel in report.accepted:
                if not (metadata_by_image.get(rel) or {}).get("caption"):
                    report.add("warning", "missing_caption", rel, "No caption yet; run scripts/generate_captions.py before preparing the dataset.")
        return report

    @staticmethod
    def _merge_duplicate_sets(sets: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        """Union overlapping duplicate sets (e.g. a curated group plus an
        automatic hash match touching one of its members) so each artwork
        is resolved exactly once. The most specific label wins."""
        priority = {"exact_duplicate": 0, "curated_duplicate": 1, "near_duplicate": 2}
        merged: list[tuple[str, set[str]]] = []
        for category, paths in sets:
            members = set(paths)
            overlapping = [m for m in merged if m[1] & members]
            for m in overlapping:
                merged.remove(m)
                members |= m[1]
                if priority[m[0]] < priority[category]:
                    category = m[0]
            merged.append((category, members))
        return [(category, sorted(members)) for category, members in merged]

    def save_report(self, report: ValidationReport, output_path: "str | Path") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Saved validation report to %s", output_path)


def check_manifest(manifest_file: "str | Path", trigger_token: str, token_counter=None, max_tokens: int = 77) -> ValidationReport:
    """Integrity check of a *prepared* training manifest -- what the trainer
    will actually read. Used on Colab, where only the packaged prepared
    dataset (not data/raw) is present. Checks every row has a readable image
    whose sides are multiples of 8, a caption that starts with the trigger
    token, fits in CLIP's 77-token window, and is not a placeholder."""
    report = ValidationReport()
    manifest_file = Path(manifest_file)
    if not manifest_file.exists():
        report.add("error", "missing_manifest", str(manifest_file), "Manifest not found; run scripts/prepare_dataset.py.")
        return report
    rows = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    report.total_images_found = len(rows)
    seen_captions: dict[str, str] = {}
    for row in rows:
        image, text = row.get("image", ""), row.get("text", "")
        problems = []
        path = Path(image)
        if not path.exists() and (manifest_file.parent / path.name).exists():
            path = manifest_file.parent / path.name
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
            if w % 8 or h % 8:
                problems.append(("bad_size", f"{w}x{h} is not a multiple of 8"))
        except Exception:
            problems.append(("unreadable_image", "image missing or unreadable"))
        if not text.startswith(trigger_token):
            problems.append(("missing_trigger", f"caption does not start with {trigger_token}"))
        if "unknown" in text or "no VLM" in text:
            problems.append(("placeholder_caption", "caption contains placeholder text"))
        if token_counter is not None and token_counter(text) > max_tokens - 2:
            problems.append(("caption_too_long", f"{token_counter(text)} tokens; CLIP truncates at {max_tokens - 2}"))
        if text in seen_captions:
            report.add("warning", "duplicate_caption", image, f"Same caption as {seen_captions[text]}")
        seen_captions.setdefault(text, image)
        if problems:
            for category, message in problems:
                report.exclude(image, category, message, severity="error")
        else:
            report.accepted.append(image)
    return report
