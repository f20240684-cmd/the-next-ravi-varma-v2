#!/usr/bin/env python3
"""Generate Ravi Varma-inspired images with Stable Diffusion 1.5 + the style LoRA.

Single prompt:
    python scripts/generate.py --prompt "<rvvarma>, royal Indian portrait, jewelry, classical oil painting" --seed 42

Prompt suite (one model load, every prompt x every seed):
    python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44

Base-model baseline for comparison (same prompts/seeds, LoRA disabled):
    python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 --no-lora \\
        --output-dir outputs/generated/baseline

Use a specific training checkpoint (checkpoint selection):
    python scripts/generate.py --prompt "..." --lora-path checkpoints/lora/ravi_varma/checkpoint-1000

Every image is saved with a sidecar .json recording prompt, seed, steps,
guidance, LoRA path/scale and device, so any result can be regenerated.
If the prompt lacks the trigger token it is prepended (unless --no-lora).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from ravi_varma.config import GenerationConfig
from ravi_varma.generation.pipeline import RaviVarmaGenerator
from ravi_varma.utils.logging import setup_logging


def load_prompt_file(path: str) -> list[dict]:
    """YAML with a `prompts:` list of {id, prompt, [negative_prompt, width, height]}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("prompts", [])
    if not entries:
        raise ValueError(f"No `prompts:` entries in {path}")
    for i, entry in enumerate(entries):
        if "prompt" not in entry:
            raise ValueError(f"Entry {i} in {path} has no `prompt`")
        entry.setdefault("id", f"prompt{i:02d}")
    return entries


def with_trigger(prompt: str, trigger: str) -> str:
    return prompt if trigger in prompt else f"{trigger}, {prompt}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Prompt text (include <rvvarma> to trigger the style; added automatically otherwise).")
    src.add_argument("--prompt-file", help="YAML prompt suite, e.g. configs/prompts/sample_prompts.yaml.")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--negative-prompt", default=None, help="Override the negative prompt from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (random if omitted).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Several seeds; each prompt is rendered once per seed.")
    parser.add_argument("--num-images", type=int, default=None, help="Images per prompt with consecutive seeds (starting at --seed).")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--lora-scale", type=float, default=None, help="Strength of the style LoRA (0 = off, 1 = as trained).")
    parser.add_argument("--lora-path", default=None, help="LoRA dir, checkpoint-<step> dir, or .safetensors file.")
    parser.add_argument("--no-lora", action="store_true", help="Base model only (baseline for comparisons).")
    parser.add_argument("--output-dir", default=None, help="Defaults to output.generated_dir in the config.")
    parser.add_argument("--skip-existing", action="store_true", help="With --prompt-file: skip prompt/seed pairs already rendered in the output dir (resume a long run).")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="[Milestone 3 preview] run the template prompt-expansion engine on the prompt first.",
    )
    parser.add_argument("--pose-image", default=None, help="[Milestone 4 preview, untested] reference image for ControlNet.")
    parser.add_argument("--controlnet-type", choices=["openpose", "canny"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--controlnet-strength", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logger = setup_logging("generate")
    config = GenerationConfig.load(args.config)
    if args.lora_path:
        config.model.lora_path = args.lora_path
    trigger = config.style.trigger_token

    entries = load_prompt_file(args.prompt_file) if args.prompt_file else [{"id": None, "prompt": args.prompt}]

    if args.seeds:
        seeds = args.seeds
    elif args.num_images:
        start = args.seed if args.seed is not None else int.from_bytes(__import__("os").urandom(3), "little")
        seeds = [start + i for i in range(args.num_images)]
    else:
        seeds = [args.seed if args.seed is not None else config.generation.seed]

    generator = RaviVarmaGenerator(config, allow_missing_lora=True, use_lora=not args.no_lora)
    control_image = None
    if args.pose_image:
        from PIL import Image

        control_image = Image.open(args.pose_image).convert("RGB")

    results = []
    started = time.time()
    for entry in entries:
        prompt = entry["prompt"]
        if args.expand:
            from ravi_varma.generation.prompt_engine import PromptExpansionEngine

            prompt = PromptExpansionEngine(trigger_token=trigger, negative_prompt=config.negative_prompt).expand(prompt).final_prompt()
            logger.info("Expanded prompt: %s", prompt)
        elif not args.no_lora:
            prompt = with_trigger(prompt, trigger)

        for seed in seeds:
            out_dir = Path(args.output_dir or config.output.generated_dir)
            if args.skip_existing and entry["id"] and seed is not None and (out_dir / f"{entry['id']}_seed{seed}.png").exists():
                logger.info("Skipping %s seed %s (already rendered)", entry["id"], seed)
                continue
            try:
                result = generator.generate(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt or entry.get("negative_prompt"),
                    seed=seed,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    width=args.width or entry.get("width"),
                    height=args.height or entry.get("height"),
                    lora_scale=args.lora_scale,
                    control_image=control_image,
                    control_type=args.controlnet_type,
                    controlnet_strength=args.controlnet_strength,
                    output_dir=args.output_dir,
                    name_prefix=entry["id"],
                    extra_metadata={"prompt_id": entry["id"]} if entry["id"] else None,
                )
            except RuntimeError as e:
                logger.error("%s", e)
                return 2
            results.append(result)
            print(f"Saved {result.image_path}  (seed {result.metadata['seed']}, lora_loaded={result.metadata['lora_loaded']})")

    if not results:
        print("Nothing to do: every prompt/seed pair already exists.")
        return 0
    if len(results) == 1:
        print(json.dumps(results[0].metadata, indent=2))
    else:
        print(f"Generated {len(results)} images in {time.time() - started:.0f}s")

    if not args.no_lora and results and not results[0].metadata["lora_loaded"]:
        print(
            "\nWARNING: no trained LoRA checkpoint was found -- these images were generated with the "
            "BASE model only and do not reflect Ravi Varma's style. Train on a GPU first (see README)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
