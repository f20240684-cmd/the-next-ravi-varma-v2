#!/usr/bin/env python3
"""The Next Ravi Varma -- Gradio web dashboard.

Usage:
    python app/app.py [--config configs/generation.yaml] [--share]

Layout and styling follow the Figma design "The Next Ravi Varma – App UI"
(app/styles.css). If a trained LoRA checkpoint does not exist yet, the app
still launches and says so in the header rather than pretending it exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ravi_varma.config import GenerationConfig
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.generation.prompt_engine import PromptExpansionEngine
from ravi_varma.utils.logging import setup_logging

from components import (
    ABOUT_MARKDOWN,
    CONTROL_MODES,
    EXAMPLES,
    SIZES,
    checkpoint_table_html,
    list_checkpoints,
    loss_chart_svg,
    lora_status_message,
    recent_images,
    run_evaluation,
    run_generation,
    run_photo_to_painting,
)

ROOT = Path(__file__).resolve().parent.parent
logger = setup_logging("app")


def _require_gradio():
    try:
        import gradio as gr

        return gr
    except ImportError:
        logger.error(
            "Gradio is not installed in this environment. Run `pip install gradio` "
            "(see requirements.txt) to launch the web dashboard. All underlying pipeline "
            "code (dataset/training/generation/evaluation) works independently of Gradio "
            "via the scripts/ CLI tools."
        )
        raise SystemExit(2)


def _theme(gr):
    """Dark 'gallery at night' palette; the same values for light and dark mode so the
    app looks as designed whatever the visitor's system setting."""
    c = dict(bg="#14100E", surface="#1E1814", surface2="#2A221C", line="#3A2F27", text="#F3E9D8",
             muted="#B5A691", gold="#C9A24B", maroon="#7A1F1F")
    settings = dict(
        body_background_fill=c["bg"], body_text_color=c["text"], body_text_color_subdued=c["muted"],
        background_fill_primary=c["surface"], background_fill_secondary=c["surface2"],
        block_background_fill=c["surface"], block_border_color=c["line"], block_label_background_fill=c["surface2"],
        block_label_text_color=c["muted"], block_title_text_color=c["text"], border_color_primary=c["line"],
        input_background_fill=c["surface2"], input_border_color=c["line"], input_border_color_focus=c["gold"],
        input_placeholder_color="#8A7C69", color_accent=c["gold"], color_accent_soft="rgba(201,162,75,0.14)",
        slider_color=c["gold"], checkbox_background_color_selected=c["gold"], checkbox_border_color_selected=c["gold"],
        button_primary_background_fill=c["maroon"], button_primary_background_fill_hover="#8E2626",
        button_primary_text_color=c["text"], button_primary_border_color="rgba(201,162,75,0.5)",
        button_secondary_background_fill="transparent", button_secondary_text_color=c["gold"],
        button_secondary_border_color="rgba(201,162,75,0.45)", button_secondary_background_fill_hover="rgba(201,162,75,0.12)",
        table_even_background_fill=c["surface"], table_odd_background_fill=c["surface2"],
    )
    import inspect

    accepted = inspect.signature(gr.themes.Base.set).parameters
    settings.update({k + "_dark": v for k, v in list(settings.items()) if k + "_dark" in accepted})
    return gr.themes.Base(
        font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    ).set(**settings)


def build_app(config_path: str):
    gr = _require_gradio()
    config = GenerationConfig.load(config_path)
    generator = RaviVarmaGenerator(config, allow_missing_lora=True)
    prompt_engine = PromptExpansionEngine(trigger_token=config.style.trigger_token, negative_prompt=config.negative_prompt)
    lora_root = config.model.lora_path or ""
    checkpoints = list_checkpoints(lora_root)
    status = lora_status_message(generator)
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")

    with gr.Blocks(css=css, theme=_theme(gr), title="The Next Ravi Varma") as demo:
        gr.HTML(
            "<div class='rv-top'><div class='rv-brand'><div class='rv-mono'>RV</div><div>"
            "<div class='rv-title'>The Next Ravi Varma</div>"
            "<div class='rv-dim'>Ravi Varma–inspired paintings from Indian epics</div></div></div>"
            f"<div class='rv-pill'><span class='rv-dot'></span>{status} · {len(checkpoints)} checkpoints</div></div>"
        )

        with gr.Tabs(elem_classes="rv-tabs"):
            # ---------------------------------------------------------- #
            with gr.Tab("Generate"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=380, elem_classes="rv-panel"):
                        gr.HTML("<div class='rv-h2'>Describe a scene</div><div class='rv-dim'>Write it plainly — "
                                "characters, place, action. We add the iconography and the trigger word.</div>")
                        scene = gr.Textbox(show_label=False, lines=3, elem_classes="rv-scene",
                                           placeholder="Damayanti talking to a swan in a palace garden")
                        with gr.Row(elem_classes="rv-chips"):
                            chips = {gr.Button(label, size="sm", elem_classes="rv-chip"): text for label, text in EXAMPLES.items()}
                        expand = gr.Checkbox(value=True, label="Expand with the epic lexicon · 27 characters · 18 places")

                        gr.HTML("<div class='rv-sec'>Style</div>")
                        style_strength = gr.Slider(0.0, 1.5, value=config.generation.lora_scale, step=0.05, label="Ravi Varma style strength")
                        lora_checkpoint = gr.Dropdown(checkpoints, value=config.model.lora_path if checkpoints else None,
                                                      label="LoRA checkpoint · saved every 250 steps")

                        gr.HTML("<div class='rv-sec'>Composition · optional <span class='rv-dim'>ControlNet</span></div>")
                        pose_image = gr.Image(label="Pose reference (a painting or photo)", type="pil", height=180, elem_classes="rv-drop")
                        control_mode = gr.Radio(list(CONTROL_MODES), value="Copy pose (OpenPose)", show_label=False, elem_classes="rv-seg")
                        controlnet_strength = gr.Slider(0.0, 1.5, value=config.controlnet.strength, step=0.05, label="Follow the reference")

                        with gr.Accordion("Advanced · seed 42 · 30 steps · CFG 7.5", open=False, elem_classes="rv-adv"):
                            seed = gr.Number(label="Seed (0 = random)", value=42, precision=0)
                            steps = gr.Slider(10, 60, value=config.generation.steps, step=1, label="Steps")
                            guidance = gr.Slider(1.0, 15.0, value=config.generation.guidance_scale, step=0.5, label="Guidance scale")
                            size = gr.Radio(list(SIZES), value="Portrait 448×640", label="Size")
                            negative_prompt = gr.Textbox(label="Negative prompt", value=config.negative_prompt, lines=3)
                        face_fix = gr.Checkbox(value=True, label="Sharpen faces · re-draws each face at 512 px (+~20 s per face)")

                        generate_btn = gr.Button("✦  Generate painting", variant="primary", elem_classes="rv-cta")
                        gr.HTML("<div class='rv-dim rv-center'>About 75 s on a Mac · 2 min with a pose reference</div>")

                    with gr.Column(scale=8, min_width=480):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5, min_width=320):
                                output_image = gr.Image(show_label=False, elem_classes="rv-frame", height=640,
                                                        show_download_button=True, interactive=False)
                                caption = gr.HTML()
                                warning = gr.Markdown()
                            with gr.Column(scale=4, min_width=280):
                                control_map = gr.Image(label="Pose map followed", height=200, interactive=False, elem_classes="rv-card-img")
                                prompt_html = gr.HTML("<div class='rv-card'><div class='rv-card-title'>Expanded prompt</div>"
                                                      "<p class='rv-dim'>Appears after you generate.</p></div>")
                                details_html = gr.HTML()
                        gr.HTML("<div class='rv-h3'>Recent paintings</div>")
                        gallery = gr.Gallery(value=recent_images(config.output.generated_dir), columns=6, height=230,
                                             show_label=False, object_fit="cover", allow_preview=True, elem_classes="rv-gallery")

                for chip, text in chips.items():
                    chip.click(lambda t=text: t, outputs=scene)
                generate_btn.click(
                    fn=lambda *a: run_generation(generator, prompt_engine, *a),
                    inputs=[scene, expand, style_strength, lora_checkpoint, pose_image, control_mode, controlnet_strength,
                            seed, steps, guidance, size, negative_prompt, face_fix],
                    outputs=[output_image, caption, prompt_html, details_html, warning, control_map, gallery],
                )

            # ---------------------------------------------------------- #
            with gr.Tab("Photo → Varma"):
                gr.HTML("<div class='rv-h1'>Photo → Ravi Varma</div><p class='rv-muted rv-lead'>Upload a photo and it is "
                        "repainted in Ravi Varma's style. Its outlines are locked (ControlNet Canny), so pose and composition "
                        "stay; strength sets how painterly it gets. Masks and modern costume are reinterpreted.</p>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes="rv-panel"):
                        photo = gr.Image(label="Your photo", type="pil", height=320)
                        photo_desc = gr.Textbox(label="Describe it (optional)", placeholder="a young woman in a sari smiling")
                        photo_strength = gr.Slider(0.3, 0.8, value=0.55, step=0.05, label="Strength · 0.5 subtle · 0.7 very painterly (outlines stay locked)")
                        photo_faces = gr.Checkbox(value=True, label="Sharpen faces (+~20 s per face)")
                        photo_seed = gr.Number(label="Seed (0 = random)", value=42, precision=0)
                        photo_btn = gr.Button("✦  Paint it like Ravi Varma", variant="primary", elem_classes="rv-cta")
                        gr.HTML("<div class='rv-dim rv-center'>About 1–2 min on a Mac (close other apps on 8 GB)</div>")
                    with gr.Column(scale=6, min_width=320):
                        photo_out = gr.Image(show_label=False, elem_classes="rv-frame", height=640, interactive=False)
                        photo_warning = gr.Markdown()
                        photo_details = gr.HTML()
                photo_btn.click(
                    fn=lambda *a: run_photo_to_painting(generator, *a),
                    inputs=[photo, photo_desc, photo_strength, photo_faces, photo_seed],
                    outputs=[photo_out, photo_details, photo_warning],
                )

            # ---------------------------------------------------------- #
            with gr.Tab("Checkpoint Hub"):
                gr.HTML("<div class='rv-h1'>Checkpoint Hub</div><p class='rv-muted rv-lead'>Seven LoRA weight files saved "
                        "while training the v3 LoRA on 182 Ravi Varma paintings. Each plugs into the same Stable Diffusion 1.5 base "
                        "— pick one in <b>Generate</b> to compare.</p>"
                        "<div class='rv-stats'>"
                        + "".join(f"<div class='rv-stat'><div class='rv-stat-v'>{v}</div><div class='rv-dim'>{l}</div></div>"
                                  for v, l in [("1,500", "optimizer steps"), ("58 min", "on a Colab T4"),
                                               ("3.2 M", "trainable parameters"), ("rank 16", "α 16 · to_q k v out")])
                        + "</div>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        gr.HTML(f"<div class='rv-card'><div class='rv-card-title'>Training loss · 50-step moving average</div>"
                                f"{loss_chart_svg(lora_root)}<p class='rv-dim'>Diffusion loss depends on the random noise level "
                                "drawn each step, so it stays noisy and flat — the images still improve. Judge checkpoints by "
                                "eye, not by this number.</p></div>")
                    with gr.Column(scale=1):
                        gr.HTML(f"<div class='rv-card'><div class='rv-card-title'>Saved checkpoints</div>{checkpoint_table_html(lora_root)}</div>")
                progression = ROOT / "outputs/evaluations/lora_vs_base/validation_progression.jpg"
                if progression.exists():
                    gr.Image(str(progression), label="Same three prompts at every checkpoint · steps 0 → 1500 (v2 run)",
                             interactive=False, show_download_button=False)

            # ---------------------------------------------------------- #
            with gr.Tab("Evaluation"):
                gr.HTML("<div class='rv-h1'>Evaluation</div><p class='rv-muted rv-lead'>Score any image against its prompt "
                        "and against the 12 held-out Ravi Varma paintings (CLIP ViT-B/32).</p>")
                with gr.Row():
                    eval_image = gr.Image(label="Generated image", type="pil", height=420)
                    with gr.Column(elem_classes="rv-panel"):
                        eval_prompt = gr.Textbox(label="Prompt used to generate it")
                        reference_dir = gr.Textbox(label="Reference corpus directory", value="data/reference")
                        eval_btn = gr.Button("Calculate metrics", variant="primary")
                        eval_output = gr.Markdown()
                eval_btn.click(fn=run_evaluation, inputs=[eval_image, eval_prompt, reference_dir], outputs=[eval_output])

            # ---------------------------------------------------------- #
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MARKDOWN, elem_classes="rv-panel rv-about")

        gr.HTML("<div class='rv-foot'><span>Stable Diffusion 1.5 + Ravi Varma LoRA (rank 16) · ControlNet OpenPose / Canny"
                f" · model {config.model.base_model}</span><span>Raja Ravi Varma (1848–1906) · source paintings public domain"
                "</span></div>")

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "generation.yaml"))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    args.config = str(Path(args.config).resolve())
    os.chdir(ROOT)  # config paths (LoRA, outputs, references) are relative to the repo root

    demo = build_app(args.config)
    demo.launch(share=args.share, server_port=args.server_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
