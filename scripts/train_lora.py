#!/usr/bin/env python3
"""Train a Ravi Varma-style LoRA on top of a Stable Diffusion base model.

Intended to run on a CUDA GPU (Colab T4 or better): see
notebooks/03_lora_training.ipynb.

Usage:
    python scripts/train_lora.py --config configs/lora_sd15.yaml
    python scripts/train_lora.py --config configs/lora_sd15.yaml --smoke-test        # 20 steps, throw-away dirs
    python scripts/train_lora.py --config configs/lora_sd15.yaml --max-train-steps 500

Training resumes automatically from the latest checkpoint in
checkpointing.output_dir (set --resume-from-checkpoint none to start over).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ravi_varma.config import LoraTrainingConfig
from ravi_varma.training.lora_trainer import LoraTrainer, check_ml_stack_available
from ravi_varma.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a lora_sd15.yaml / lora_sdxl.yaml file.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Override training.max_train_steps (useful for a quick smoke test).")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Override checkpointing.resume_from_checkpoint ('latest', a path, or 'none').")
    parser.add_argument("--output-dir", default=None, help="Override checkpointing.output_dir.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="20 optimizer steps with a checkpoint + validation at step 10, written to outputs/smoke_test/ "
        "(never touches or resumes the real run).",
    )
    args = parser.parse_args()

    logger = setup_logging("train_lora")

    config = LoraTrainingConfig.load(args.config)
    if args.max_train_steps is not None:
        config.training.max_train_steps = args.max_train_steps
    if args.resume_from_checkpoint is not None:
        config.checkpointing.resume_from_checkpoint = None if args.resume_from_checkpoint.lower() == "none" else args.resume_from_checkpoint
    if args.output_dir is not None:
        config.checkpointing.output_dir = args.output_dir
    if args.smoke_test:
        config.training.max_train_steps = args.max_train_steps or 20
        config.optimizer.lr_warmup_steps = 2
        config.checkpointing.output_dir = "outputs/smoke_test/lora"
        config.checkpointing.checkpointing_steps = 10
        config.checkpointing.resume_from_checkpoint = None
        config.logging.logging_dir = "outputs/smoke_test/logs"
        config.logging.log_every_n_steps = 5
        config.validation.validation_steps = 10
        config.validation.validation_prompts = config.validation.prompts()[:1]
        config.validation.validation_prompt = None
        config.validation.run_at_start = False
        config.validation.num_inference_steps = 15

    missing = check_ml_stack_available()
    if missing:
        logger.error(missing)
        logger.error(
            "This machine cannot run real LoRA training. Use Google Colab "
            "(notebooks/03_lora_training.ipynb) or a CUDA machine with requirements.txt installed."
        )
        return 2

    trainer = LoraTrainer(config)
    try:
        output_dir = trainer.train()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    logger.info("Training complete. LoRA checkpoint saved to %s", output_dir)
    logger.info("Next: python scripts/generate.py --prompt \"...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
