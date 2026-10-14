#!/usr/bin/env python3
"""Milestone 4 evaluation: does ControlNet keep a reference pose, and what does
it cost in style / prompt adherence?

Pose references are held-out Ravi Varma paintings (never seen in training).
Each is rendered with its own caption and seed 42 three ways -- LoRA only,
LoRA + OpenPose, LoRA + Canny -- and scored on:
  * pose_pck        -- fraction of the reference's body joints reproduced (OpenPose on both; see pose_score.py)
  * style_ref_mean  -- mean CLIP image similarity to the OTHER held-out paintings (its own source excluded,
                       otherwise Canny, which traces the source, would score itself)
  * prompt_adherence-- CLIP similarity to the caption (trigger removed)

    python scripts/eval_controlnet.py render --variant lora      # one variant per process (8 GB Macs)
    python scripts/eval_controlnet.py render --variant openpose
    python scripts/eval_controlnet.py render --variant canny
    python scripts/eval_controlnet.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REFS = ["0225.jpg", "0189.jpg", "0212.jpg", "0236.jpg", "0026.jpg", "0173.jpg"]
VARIANTS = ["lora", "openpose", "canny"]
OUT = Path("outputs/generated/m4_controlnet")
EVAL = Path("outputs/evaluations/m4_controlnet")


def references() -> list[dict]:
    rows = {Path(r["image"]).name: r for r in map(json.loads, Path("data/reference/reference.jsonl").read_text().splitlines())}
    return [rows[name] for name in REFS]


def render(variant: str, seed: int) -> None:
    from PIL import Image

    from ravi_varma.config import GenerationConfig
    from ravi_varma.generation.pipeline import RaviVarmaGenerator

    generator = RaviVarmaGenerator(GenerationConfig.load("configs/generation.yaml"), allow_missing_lora=False)
    for ref in references():
        stem = Path(ref["image"]).stem
        if (OUT / variant / f"{stem}_seed{seed}.png").exists():
            continue
        result = generator.generate(
            prompt=ref["text"],
            seed=seed,
            width=ref["width"],
            height=ref["height"],
            control_image=None if variant == "lora" else Image.open(ref["image"]).convert("RGB"),
            control_type=None if variant == "lora" else variant,
            output_dir=OUT / variant,
            name_prefix=stem,
            extra_metadata={"pose_reference": ref["image"], "reference_title": ref["title"]},
        )
        print("Saved", result.image_path, flush=True)


def report(seed: int) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    from ravi_varma.evaluation.clip_score import embed_images, embed_texts
    from ravi_varma.evaluation.pose_score import pose_keypoints, pose_pck

    refs = references()
    all_refs = sorted(Path("data/reference").glob("*.jpg"))
    ref_emb = embed_images(all_refs)
    rows = []
    for ref in refs:
        stem = Path(ref["image"]).stem
        size = (ref["width"], ref["height"])
        ref_img = ImageOps.fit(Image.open(ref["image"]).convert("RGB"), size)  # same crop the generator used
        ref_pose = pose_keypoints(ref_img)
        others = np.array([p.name != Path(ref["image"]).name for p in all_refs])
        row = {"reference": ref["image"], "title": ref["title"], "people_in_reference": len(ref_pose)}
        for v in VARIANTS:
            path = OUT / v / f"{stem}_seed{seed}.png"
            img = Image.open(path).convert("RGB")
            pck = pose_pck(ref_pose, pose_keypoints(img))
            emb = embed_images([img])[0]
            row[v] = {
                "pose_pck": None if pck is None else round(pck, 3),
                "style_ref_mean": round(float((ref_emb[others] @ emb).mean()), 4),
                "prompt_adherence": round(float(embed_texts([ref["text"]])[0] @ emb), 4),
            }
        rows.append(row)
        print(stem, {v: row[v] for v in VARIANTS}, flush=True)

    def mean(v, k):
        vals = [r[v][k] for r in rows if r[v][k] is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {v: {k: mean(v, k) for k in ("pose_pck", "style_ref_mean", "prompt_adherence")} for v in VARIANTS}
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "summary.json").write_text(json.dumps({
        "seed": seed,
        "pose_pck_threshold": 0.1,
        "notes": "pose_pck: share of reference body joints within 0.1 (normalised) of the generated pose; "
                 "style_ref_mean excludes the painting used as the pose reference.",
        "mean": summary, "per_reference": rows}, indent=2))

    # Grid: reference | pose map | LoRA only | + OpenPose | + Canny
    cell_w, cell_h, pad = 220, 300, 30
    cols = ["Reference (held out)", "OpenPose map", "LoRA only", "LoRA + OpenPose", "LoRA + Canny"]
    grid = Image.new("RGB", (cell_w * len(cols), pad + cell_h * len(rows)), "white")
    d = ImageDraw.Draw(grid)
    for c, title in enumerate(cols):
        d.text((c * cell_w + 6, 8), title, fill="black")
    for r, ref in enumerate(refs):
        stem = Path(ref["image"]).stem
        cells = [Image.open(ref["image"]), Image.open(OUT / "openpose" / f"{stem}_seed{seed}_control.png")]
        cells += [Image.open(OUT / v / f"{stem}_seed{seed}.png") for v in VARIANTS]
        for c, im in enumerate(cells):
            im = ImageOps.contain(im.convert("RGB"), (cell_w - 6, cell_h - 6))
            grid.paste(im, (c * cell_w + (cell_w - im.width) // 2, pad + r * cell_h + (cell_h - im.height) // 2))
    grid.save(EVAL / "grid.jpg", quality=88)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["render", "report"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "render":
        if not args.variant:
            parser.error("render needs --variant")
        render(args.variant, args.seed)
    else:
        report(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
