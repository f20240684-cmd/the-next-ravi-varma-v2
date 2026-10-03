"""Run-level evaluation of a set of generated images (the Milestone-2 scope:
a simple, credible check of the style LoRA -- not the full multimodal
evaluation framework planned for Milestone 4).

Given a directory produced by `scripts/generate.py` (images + sidecar JSON),
this computes, per image:

  * prompt_adherence  -- CLIP ViT-B/32 cosine similarity between the image
                         and its prompt (trigger token removed)
  * style_ref_mean    -- mean CLIP image similarity to the HELD-OUT reference
                         paintings (data/reference, never trained on)
  * style_ref_top3    -- mean of the 3 most similar references
  * nearest_train_sim -- similarity to the closest TRAINING image; values
                         close to 1.0 flag possible memorisation/copying

and per prompt (across seeds):

  * diversity         -- mean pairwise CLIP distance (1 - cosine) between
                         images of the same prompt; ~0 means near-identical
                         outputs regardless of seed (mode collapse)

If a baseline directory (same prompt ids + seeds, rendered with --no-lora)
is given, paired deltas LoRA - base are reported for every metric. The
paired comparison is the most informative number here: absolute CLIP
similarities are not calibrated and mean little on their own.

Visual quality and anatomy (faces, hands, limbs) have no reliable automatic
metric, so the evaluator writes a contact sheet plus a `qualitative_review.csv`
with empty 1-5 rubric columns for a human reviewer. No score is ever
fabricated: if CLIP is unavailable every automatic metric is None.
"""
from __future__ import annotations

import csv
import itertools
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from ravi_varma.evaluation.clip_score import embed_images, embed_texts
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

METRIC_NOTES = {
    "prompt_adherence": (
        "CLIP ViT-B/32 image-text cosine similarity with the trigger token removed. Measures coarse "
        "semantic agreement (subjects, setting); blind to counts, hands, exact poses. Typical range 0.20-0.35."
    ),
    "style_ref_mean": (
        "Mean CLIP image-image similarity to held-out Ravi Varma paintings. Mixes content and style "
        "(a palace scene scores higher against palace references), so compare LoRA vs base on the "
        "same prompts/seeds rather than reading absolute values."
    ),
    "style_ref_top3": "Mean similarity to the 3 closest held-out references; less diluted by unrelated subjects.",
    "nearest_train_sim": (
        "Similarity to the closest training image. Values above ~0.95 suggest the model is reproducing "
        "a training painting rather than composing a new one."
    ),
    "diversity": (
        "Mean pairwise (1 - cosine) CLIP distance between images of the same prompt across seeds. "
        "Very low values mean the seed barely changes the output."
    ),
    "human_rubric": (
        "visual_quality, anatomy, style_consistency and prompt_adherence are rated 1-5 by a person "
        "in qualitative_review.csv; anatomy covers faces, eyes, hands/fingers and limb count."
    ),
}


@dataclass
class ImageRecord:
    image: str
    prompt_id: Optional[str]
    seed: Optional[int]
    prompt: Optional[str]
    lora_loaded: Optional[bool]
    lora_scale: Optional[float]
    prompt_adherence: Optional[float] = None
    style_ref_mean: Optional[float] = None
    style_ref_top3: Optional[float] = None
    nearest_reference: Optional[str] = None
    nearest_train_sim: Optional[float] = None
    nearest_train_image: Optional[str] = None


@dataclass
class RunEvaluation:
    input_dir: str
    num_images: int
    num_references: int
    num_training_images: int
    images: list[ImageRecord] = field(default_factory=list)
    per_prompt: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    baseline_comparison: Optional[dict] = None
    notes: dict = field(default_factory=lambda: dict(METRIC_NOTES))


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("_"))


def _sidecar(path: Path) -> dict:
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _reference_images(reference_dir: Path) -> list[Path]:
    return _list_images(reference_dir)


def _training_images(train_manifest: Optional[Path]) -> list[Path]:
    if not train_manifest or not Path(train_manifest).exists():
        return []
    paths = []
    with Path(train_manifest).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = Path(json.loads(line)["image"])
                if p.exists():
                    paths.append(p)
    return paths


def _score_directory(images: list[Path], ref_emb, ref_paths, train_emb, train_paths, device: str) -> list[ImageRecord]:
    import numpy as np

    records = []
    sidecars = [_sidecar(p) for p in images]
    for p, side in zip(images, sidecars):
        records.append(
            ImageRecord(
                image=str(p),
                prompt_id=side.get("prompt_id"),
                seed=side.get("seed"),
                prompt=side.get("prompt"),
                lora_loaded=side.get("lora_loaded"),
                lora_scale=side.get("lora_scale"),
            )
        )
    img_emb = embed_images(images, device=device)
    if img_emb is None:
        logger.warning("CLIP unavailable: automatic metrics left empty (nothing fabricated).")
        return records

    prompts = [r.prompt for r in records]
    with_prompt = [i for i, t in enumerate(prompts) if t]
    if with_prompt:
        txt_emb = embed_texts([prompts[i] for i in with_prompt], device=device)
        for k, i in enumerate(with_prompt):
            records[i].prompt_adherence = round(float(img_emb[i] @ txt_emb[k]), 4)

    if ref_emb is not None and len(ref_paths):
        sims = img_emb @ ref_emb.T
        for i, rec in enumerate(records):
            row = np.sort(sims[i])[::-1]
            rec.style_ref_mean = round(float(sims[i].mean()), 4)
            rec.style_ref_top3 = round(float(row[:3].mean()), 4)
            rec.nearest_reference = ref_paths[int(sims[i].argmax())].name

    if train_emb is not None and len(train_paths):
        sims = img_emb @ train_emb.T
        for i, rec in enumerate(records):
            j = int(sims[i].argmax())
            rec.nearest_train_sim = round(float(sims[i, j]), 4)
            rec.nearest_train_image = train_paths[j].name

    # Keep embeddings for diversity computation.
    for rec, emb in zip(records, img_emb):
        rec._emb = emb  # type: ignore[attr-defined]
    return records


def _per_prompt(records: list[ImageRecord]) -> dict:
    groups: dict[str, list[ImageRecord]] = {}
    for r in records:
        groups.setdefault(r.prompt_id or Path(r.image).stem, []).append(r)
    out = {}
    for pid, recs in sorted(groups.items()):
        embs = [getattr(r, "_emb", None) for r in recs]
        diversity = None
        if len(recs) > 1 and all(e is not None for e in embs):
            diversity = round(statistics.mean(1 - float(a @ b) for a, b in itertools.combinations(embs, 2)), 4)
        out[pid] = {
            "num_images": len(recs),
            "prompt_adherence": _mean([r.prompt_adherence for r in recs]),
            "style_ref_mean": _mean([r.style_ref_mean for r in recs]),
            "style_ref_top3": _mean([r.style_ref_top3 for r in recs]),
            "nearest_train_sim_max": max((r.nearest_train_sim for r in recs if r.nearest_train_sim is not None), default=None),
            "diversity": diversity,
        }
    return out


def _summary(records: list[ImageRecord], per_prompt: dict) -> dict:
    return {
        "prompt_adherence": _mean([r.prompt_adherence for r in records]),
        "style_ref_mean": _mean([r.style_ref_mean for r in records]),
        "style_ref_top3": _mean([r.style_ref_top3 for r in records]),
        "nearest_train_sim_max": max((r.nearest_train_sim for r in records if r.nearest_train_sim is not None), default=None),
        "possible_memorisation": [Path(r.image).name for r in records if (r.nearest_train_sim or 0) > 0.95],
        "diversity": _mean([v["diversity"] for v in per_prompt.values()]),
    }


def _paired(lora: list[ImageRecord], base: list[ImageRecord]) -> Optional[dict]:
    base_by_key = {(r.prompt_id, r.seed): r for r in base}
    pairs = [(r, base_by_key[(r.prompt_id, r.seed)]) for r in lora if (r.prompt_id, r.seed) in base_by_key]
    if not pairs:
        return None
    out = {"num_pairs": len(pairs)}
    for metric in ("prompt_adherence", "style_ref_mean", "style_ref_top3", "nearest_train_sim"):
        deltas = [getattr(a, metric) - getattr(b, metric) for a, b in pairs if getattr(a, metric) is not None and getattr(b, metric) is not None]
        if deltas:
            out[metric] = {
                "lora_mean": _mean([getattr(a, metric) for a, _ in pairs]),
                "base_mean": _mean([getattr(b, metric) for _, b in pairs]),
                "mean_delta": round(statistics.mean(deltas), 4),
                "lora_higher_in": f"{sum(d > 0 for d in deltas)}/{len(deltas)} pairs",
            }
    return out


def contact_sheet(images: list[Path], out_path: Path, labels: Optional[list[str]] = None, cell: int = 256, cols: int = 5) -> Path:
    rows = max(1, (len(images) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, p in enumerate(images):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell))
        x, y = (k % cols) * cell, (k // cols) * (cell + 18)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        draw.text((x + 4, y + cell + 3), (labels[k] if labels else p.stem)[:40], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path


def evaluate_run(
    input_dir: "str | Path",
    reference_dir: "str | Path",
    output_dir: "str | Path",
    train_manifest: "str | Path | None" = None,
    baseline_dir: "str | Path | None" = None,
    device: str = "cpu",
) -> RunEvaluation:
    input_dir, reference_dir, output_dir = Path(input_dir), Path(reference_dir), Path(output_dir)
    images = _list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No generated images found in {input_dir}")

    ref_paths = _reference_images(reference_dir)
    train_paths = _training_images(Path(train_manifest) if train_manifest else None)
    if not ref_paths:
        logger.warning("No reference images in %s; style metrics will be empty. Run prepare_dataset.py (curation.holdout).", reference_dir)
    ref_emb = embed_images(ref_paths, device=device) if ref_paths else None
    train_emb = embed_images(train_paths, device=device) if train_paths else None

    records = _score_directory(images, ref_emb, ref_paths, train_emb, train_paths, device)
    per_prompt = _per_prompt(records)
    run = RunEvaluation(
        input_dir=str(input_dir),
        num_images=len(records),
        num_references=len(ref_paths),
        num_training_images=len(train_paths),
        images=records,
        per_prompt=per_prompt,
        summary=_summary(records, per_prompt),
    )

    base_records: list[ImageRecord] = []
    if baseline_dir:
        base_images = _list_images(Path(baseline_dir))
        if base_images:
            base_records = _score_directory(base_images, ref_emb, ref_paths, train_emb, train_paths, device)
            base_pp = _per_prompt(base_records)
            run.baseline_comparison = _paired(records, base_records) or {}
            run.baseline_comparison["baseline_dir"] = str(baseline_dir)
            run.baseline_comparison["diversity"] = {
                "lora_mean": run.summary["diversity"],
                "base_mean": _summary(base_records, base_pp)["diversity"],
            }
        else:
            logger.warning("Baseline directory %s has no images; skipping paired comparison.", baseline_dir)

    _write_outputs(run, records, base_records, output_dir)
    return run


def _write_outputs(run: RunEvaluation, records: list[ImageRecord], base_records: list[ImageRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(ImageRecord.__dataclass_fields__.keys())

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["set"] + fields)
        writer.writeheader()
        for name, recs in (("lora", records), ("baseline", base_records)):
            for r in recs:
                writer.writerow({"set": name, **{k: v for k, v in asdict(r).items() if k in fields}})

    payload = {k: v for k, v in asdict(run).items() if k != "images"}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human review template: one row per image, empty 1-5 rubric columns.
    review = output_dir / "qualitative_review.csv"
    if not review.exists():  # never overwrite a reviewer's scores
        with review.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "set", "prompt_id", "seed", "visual_quality_1to5", "anatomy_1to5", "style_consistency_1to5", "prompt_adherence_1to5", "notes"])
            for name, recs in (("lora", records), ("baseline", base_records)):
                for r in recs:
                    writer.writerow([Path(r.image).name, name, r.prompt_id, r.seed, "", "", "", "", ""])

    contact_sheet([Path(r.image) for r in records], output_dir / "contact_sheet.jpg",
                  labels=[f"{r.prompt_id or Path(r.image).stem} s{r.seed}" for r in records])
    if base_records:
        contact_sheet([Path(r.image) for r in base_records], output_dir / "contact_sheet_baseline.jpg",
                      labels=[f"BASE {r.prompt_id or Path(r.image).stem} s{r.seed}" for r in base_records])
    logger.info("Wrote evaluation -> %s (metrics.csv, summary.json, qualitative_review.csv, contact sheets)", output_dir)
