"""Helper functions used by app/app.py -- kept separate so the Gradio wiring
in app.py stays short and readable."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from PIL import Image

from ravi_varma.config import GenerationConfig
from ravi_varma.evaluation.clip_score import clip_available, text_image_similarity
from ravi_varma.evaluation.style_similarity import compute_style_similarity
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.generation.prompt_engine import PromptExpansionEngine

ABOUT_MARKDOWN = """
## About "The Next Ravi Varma"

**Objective.** Inspired by ["The Next Rembrandt"](https://www.nextrembrandt.com/) (2016), this
project applies modern text-to-image diffusion + LoRA fine-tuning to Raja Ravi Varma's visual
vocabulary, letting you generate *novel* digital paintings of scenes from Indian mythology and
history in a Ravi Varma-inspired style.

**Raja Ravi Varma (1848-1906)** was a pioneering Indian painter known for blending European
academic realism with Indian mythological and historical subject matter; his original works are
in the public domain.

**Pipeline.** Dataset curation -> VLM captioning -> LoRA fine-tuning (PEFT) on a Stable Diffusion
base model -> prompt expansion -> optional ControlNet pose/edge conditioning -> generation ->
CLIP-based evaluation.

**Limitations.**
- The LoRA is trained on a limited image corpus; it approximates *some* stylistic tendencies, not
  a perfect reproduction of the artist's technique.
- Evaluation metrics (CLIP alignment, embedding-based style similarity) are rough, comparative
  signals -- **not** proof of artistic authenticity or quality.
- Diffusion models can still produce anatomical errors (hands, limbs) despite the negative prompt.
- This tool cannot and does not claim to produce genuine Raja Ravi Varma paintings.

**Responsible use.** All output here is AI-generated and should be labeled as such if shared.
See the README's "Ethical & Copyright Considerations" section for dataset provenance requirements.
"""


def list_checkpoints(lora_root: "str | Path") -> list[tuple[str, str]]:
    """(label, path) for the final LoRA and every checkpoint-<step>, labelled with the
    mean training loss of the 50 steps before it (from loss_history.csv)."""
    root = Path(lora_root)
    losses = {}
    history = root / "loss_history.csv"
    if history.exists():
        with history.open() as f:
            losses = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(f)}

    def label(step: int) -> str:
        window = [losses[s] for s in range(step - 49, step + 1) if s in losses]
        return f"step {step}" + (f" · loss {sum(window) / len(window):.3f}" if window else "")

    found = []
    for ckpt in root.glob("checkpoint-*"):
        if ckpt.name.split("-")[-1].isdigit() and (ckpt / "pytorch_lora_weights.safetensors").exists():
            found.append((int(ckpt.name.split("-")[-1]), str(ckpt)))
    found.sort()
    options = [(label(step), path) for step, path in found]
    if (root / "pytorch_lora_weights.safetensors").exists():
        final_step = max(losses) if losses else (found[-1][0] if found else 0)
        options.append((f"final ({label(final_step)})", str(root)))
    return options


def checkpoint_hub_markdown(lora_root: "str | Path") -> str:
    rows = list_checkpoints(lora_root)
    if not rows:
        return f"No LoRA checkpoints found in `{lora_root}`. Train first (see README)."
    lines = ["| Checkpoint | Mean loss (last 50 steps) | Size | Path |", "|---|---|---|---|"]
    for name, path in rows:
        weights = Path(path) / "pytorch_lora_weights.safetensors"
        loss = name.split("loss ")[-1].rstrip(")") if "loss" in name else "–"
        lines.append(f"| {name.split(' (')[0].split(' ·')[0]} | {loss} | {weights.stat().st_size / 1e6:.1f} MB | `{path}` |")
    lines.append("")
    lines.append("Diffusion loss is noisy (it depends on the random timestep drawn), so a lower number "
                 "does not mean a better-looking checkpoint -- compare them visually with the selector in *Generate*.")
    return "\n".join(lines)


def format_metadata(metadata: dict) -> str:
    return "```json\n" + json.dumps(metadata, indent=2) + "\n```"


def lora_status_message(generator: RaviVarmaGenerator) -> str:
    try:
        generator.load()
    except RuntimeError as e:
        return f"⚠️ {e}"
    if generator.lora_loaded:
        return "✅ Ravi Varma LoRA checkpoint loaded."
    return (
        "⚠️ No trained LoRA checkpoint found at the configured `model.lora_path`. "
        "Generations will use the base Stable Diffusion model only, without Ravi Varma styling. "
        "Run `python scripts/train_lora.py --config configs/lora_sd15.yaml` first."
    )


def run_generation(
    generator: RaviVarmaGenerator,
    prompt_engine: PromptExpansionEngine,
    scene_description: str,
    style_strength: float,
    seed: Optional[int],
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    negative_prompt: str,
    pose_image: Optional[Image.Image],
    controlnet_type: Optional[str],
    controlnet_strength: float,
    controlnet_enabled: bool,
    lora_checkpoint: Optional[str] = None,
):
    """Generate-tab logic -- returns (image, expanded_prompt_text, metadata_markdown,
    warning_markdown, control_map_or_None)."""
    if not scene_description or not scene_description.strip():
        return None, "", "", "⚠️ Please enter a scene description.", None

    seed = int(seed) if seed else None  # Gradio sends 0/None for "random"
    if lora_checkpoint and lora_checkpoint != generator.config.model.lora_path:
        generator.set_lora(lora_checkpoint)

    expanded = prompt_engine.expand(scene_description, enable_style=style_strength > 0)
    final_prompt = expanded.final_prompt()

    control_image = pose_image if (controlnet_enabled and pose_image is not None) else None

    try:
        result = generator.generate(
            prompt=final_prompt,
            negative_prompt=negative_prompt or expanded.negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            lora_scale=style_strength,
            control_image=control_image,
            control_type=controlnet_type,
            controlnet_strength=controlnet_strength,
        )
    except RuntimeError as e:
        return None, final_prompt, "", f"⚠️ {e}", None

    warning = "" if result.metadata["lora_loaded"] else (
        "⚠️ Generated with the BASE model -- no LoRA checkpoint was found, so this image does "
        "not reflect Ravi Varma's style."
    )
    return result.image, final_prompt, format_metadata(result.metadata), warning, result.control_image


def run_evaluation(image: Optional[Image.Image], prompt: str, reference_dir: str):
    if image is None:
        return "⚠️ Upload or select a generated image first."
    availability = clip_available()
    if not availability.available:
        return f"⚠️ CLIP is unavailable in this environment ({availability.reason}). No score was fabricated."

    text_score = text_image_similarity(image, prompt) if prompt else None
    style = compute_style_similarity(image, reference_dir)

    lines = ["### Evaluation results", ""]
    lines.append(f"- **Text-image alignment (CLIP):** {text_score:.4f}" if text_score is not None else "- **Text-image alignment:** N/A (no prompt provided)")
    if style.num_references_used:
        lines.append(f"- **Style similarity (mean over {style.num_references_used} refs):** {style.mean_similarity:.4f}")
        lines.append(f"- **Style similarity (median):** {style.median_similarity:.4f}")
        lines.append(f"- **Style similarity (nearest reference):** {style.nearest_similarity:.4f}")
    else:
        lines.append(f"- **Style similarity:** N/A (no reference images found in `{reference_dir}`)")
    lines.append("")
    lines.append(f"*{style.caveat}*")
    return "\n".join(lines)
