"""Helper functions used by app/app.py -- kept separate so the Gradio wiring
in app.py stays short and readable. HTML fragments use the classes defined in
app/styles.css (design: Figma file "The Next Ravi Varma – App UI")."""
from __future__ import annotations

import csv
import html
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from ravi_varma.evaluation.clip_score import clip_available, text_image_similarity
from ravi_varma.evaluation.style_similarity import compute_style_similarity
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.generation.prompt_engine import PromptExpansionEngine, _default_token_counter

SIZES = {"Portrait 448×640": (448, 640), "Landscape 640×448": (640, 448), "Square 512×512": (512, 512)}
CONTROL_MODES = {"Copy pose (OpenPose)": "openpose", "Trace edges (Canny)": "canny"}
EXAMPLES = {  # chip label -> scene it fills in
    "Damayanti and the swan": "Damayanti talking to a swan in a palace garden",
    "Krishna and Arjuna": "Krishna guiding Arjuna in his chariot",
    "Shakuntala's letter": "Shakuntala writing a letter in a forest",
}

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


def _loss_history(lora_root: "str | Path") -> dict[int, float]:
    history = Path(lora_root) / "loss_history.csv"
    if not history.exists():
        return {}
    with history.open() as f:
        return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(f)}


def _mean_loss(losses: dict[int, float], step: int, window: int = 50) -> Optional[float]:
    vals = [losses[s] for s in range(step - window + 1, step + 1) if s in losses]
    return sum(vals) / len(vals) if vals else None


def list_checkpoints(lora_root: "str | Path") -> list[tuple[str, str]]:
    """(label, path) for the final LoRA and every checkpoint-<step>, labelled with the
    mean training loss of the 50 steps before it (from loss_history.csv)."""
    root = Path(lora_root)
    losses = _loss_history(root)

    def label(step: int) -> str:
        loss = _mean_loss(losses, step)
        return f"step {step}" + (f" · loss {loss:.3f}" if loss is not None else "")

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


def checkpoint_table_html(lora_root: "str | Path") -> str:
    rows = list_checkpoints(lora_root)
    if not rows:
        return f"<p class='rv-dim'>No LoRA checkpoints found in <code>{html.escape(str(lora_root))}</code>. Train first (see README).</p>"
    out = ["<table class='rv-table'><tr><th>Checkpoint</th><th>Mean loss</th><th>Size</th><th></th></tr>"]
    for name, path in rows:
        final = name.startswith("final")
        loss = name.split("loss ")[-1].rstrip(")") if "loss" in name else "–"
        size = (Path(path) / "pytorch_lora_weights.safetensors").stat().st_size / 1e6
        title = "Final" if final else name.split(" ·")[0].capitalize()
        tag = "<span class='rv-badge'>Default</span>" if final else "<span class='rv-gold'>select in Generate</span>"
        out.append(f"<tr class='{'rv-hi' if final else ''}'><td><b>{title}</b></td><td>{loss}</td><td>{size:.1f} MB</td><td class='rv-right'>{tag}</td></tr>")
    out.append("</table>")
    return "".join(out)


def loss_chart_svg(lora_root: "str | Path", checkpoints=(250, 500, 750, 1000, 1250, 1500)) -> str:
    """Inline SVG of the 50-step moving-average training loss, checkpoints marked."""
    losses = _loss_history(lora_root)
    if not losses:
        return "<p class='rv-dim'>No loss_history.csv found.</p>"
    last = max(losses)
    pts = [(s, _mean_loss(losses, s)) for s in range(50, last + 1, 25)]
    lo, hi = min(v for _, v in pts) - 0.01, max(v for _, v in pts) + 0.01
    W, H, x0, y0, b = 580, 230, 42, 10, 28
    X = lambda s: x0 + s / last * (W - x0 - 10)
    Y = lambda v: y0 + (1 - (v - lo) / (hi - lo)) * (H - y0 - b)
    parts = [f"<svg viewBox='0 0 {W} {H}' class='rv-chart' role='img' aria-label='Training loss curve'>"]
    for g in (lo + 0.01, (lo + hi) / 2, hi - 0.01):
        parts.append(f"<line x1='{x0}' x2='{W - 10}' y1='{Y(g):.1f}' y2='{Y(g):.1f}' class='grid'/><text x='0' y='{Y(g) + 4:.1f}'>{g:.2f}</text>")
    parts.append("<polyline class='line' points='" + " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in pts) + "'/>")
    for s in checkpoints:
        v = _mean_loss(losses, s)
        if v is None:
            continue
        parts.append(f"<line x1='{X(s):.1f}' x2='{X(s):.1f}' y1='{Y(v):.1f}' y2='{H - b + 4}' class='tick'/>"
                     f"<circle cx='{X(s):.1f}' cy='{Y(v):.1f}' r='4.5' class='dot'/>"
                     f"<text x='{X(s):.1f}' y='{H - 8}' text-anchor='middle'>{s}</text>")
    parts.append("</svg>")
    return "".join(parts)


def recent_images(directory: "str | Path", n: int = 6) -> list[str]:
    files = [p for p in Path(directory).glob("*.png") if not p.stem.endswith("_control")]
    return [str(p) for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:n]]


def lora_status_message(generator: RaviVarmaGenerator) -> str:
    try:
        generator.load()
    except RuntimeError as e:
        return f"⚠️ {e}"
    if generator.lora_loaded:
        return "LoRA loaded"
    return "No LoRA found — base model only"


def _card(title: str, body: str) -> str:
    return f"<div class='rv-card'><div class='rv-card-title'>{title}</div>{body}</div>"


def run_generation(
    generator: RaviVarmaGenerator,
    prompt_engine: PromptExpansionEngine,
    scene_description: str,
    expand: bool,
    style_strength: float,
    lora_checkpoint: Optional[str],
    pose_image: Optional[Image.Image],
    control_mode: str,
    controlnet_strength: float,
    seed: Optional[float],
    steps: int,
    guidance_scale: float,
    size: str,
    negative_prompt: str,
    face_fix: bool = True,
):
    """Generate-tab logic. Returns (image, caption_html, prompt_html, details_html,
    warning_markdown, control_map_or_None, recent_gallery)."""
    out_dir = generator.config.output.generated_dir
    if not scene_description or not scene_description.strip():
        return None, "", "", "", "⚠️ Please describe a scene first.", None, recent_images(out_dir)

    seed = int(seed) if seed else None  # Gradio sends 0/None for "random"
    if lora_checkpoint and lora_checkpoint != generator.config.model.lora_path:
        generator.set_lora(lora_checkpoint)

    scene = scene_description.strip()
    trigger = prompt_engine.trigger_token if style_strength > 0 else None
    if expand:
        expanded = prompt_engine.expand(scene, enable_style=style_strength > 0)
        final_prompt = expanded.final_prompt()
    else:
        final_prompt = scene if (not trigger or trigger in scene) else f"{trigger}, {scene}"
    width, height = SIZES.get(size, (448, 640))

    started = time.perf_counter()
    try:
        result = generator.generate(
            prompt=final_prompt,
            negative_prompt=negative_prompt or None,
            seed=seed,
            steps=int(steps),
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            lora_scale=style_strength,
            face_fix=face_fix,
            control_image=pose_image,
            control_type=CONTROL_MODES.get(control_mode, "openpose") if pose_image is not None else None,
            controlnet_strength=controlnet_strength,
            extra_metadata={"scene": scene},
        )
    except RuntimeError as e:
        return None, "", "", "", f"⚠️ {e}", None, recent_images(out_dir)
    seconds = time.perf_counter() - started
    m = result.metadata

    caption = (f"<div class='rv-caption'>{html.escape(scene)}</div>"
               "<div class='rv-dim'>AI-generated · Ravi Varma–inspired, not an authentic painting</div>")
    tokens = _default_token_counter()(final_prompt)
    prompt_html = _card("Expanded prompt" if expand else "Prompt",
                        f"<p class='rv-muted'>{html.escape(final_prompt)}</p><p class='rv-gold'>{tokens} / 75 CLIP tokens</p>")
    lora = "off" if not m["lora_loaded"] else (
        (Path(m["lora_weights"]).parent.name.replace("checkpoint-", "step ") if Path(m["lora_weights"]).parent.name.startswith("checkpoint-") else "final")
        + f" · scale {m['lora_scale']:.2f}")
    rows = [("Seed", m["seed"]), ("Steps · guidance", f"{m['steps']} · {m['guidance_scale']}"), ("LoRA", lora),
            ("Pose control", f"{m['controlnet_type']} · {m['controlnet_strength']}" if m["controlnet_enabled"] else "none"),
            ("Faces sharpened", m["faces_fixed"] if m.get("faces_fixed") is not None else "off"),
            ("Decoder", (m.get("vae") or "SD1.5 default").split("/")[-1]), ("Time", f"{seconds:.0f} s · {m['device']}"),
            ("Saved as", Path(str(result.image_path)).name if result.image_path else "–")]
    details = _card("Details", "".join(f"<div class='rv-kv'><span>{k}</span><b>{html.escape(str(v))}</b></div>" for k, v in rows))
    warning = "" if m["lora_loaded"] else "⚠️ Generated with the **base model** — no LoRA checkpoint was found."
    return result.image, caption, prompt_html, details, warning, result.control_image, recent_images(out_dir)


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
