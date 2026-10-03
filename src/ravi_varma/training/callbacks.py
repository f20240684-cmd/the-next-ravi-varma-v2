"""Training-time callbacks: checkpoint save/resume bookkeeping and periodic
validation-image generation. Kept separate from `lora_trainer.py` so the
training loop itself stays readable.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Saves numbered checkpoints under `output_dir/checkpoint-<step>` and
    prunes old ones beyond `checkpoints_total_limit`."""

    def __init__(self, output_dir: "str | Path", checkpoints_total_limit: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_total_limit = checkpoints_total_limit

    def checkpoint_dirs(self) -> list[Path]:
        dirs = [p for p in self.output_dir.glob("checkpoint-*") if p.is_dir()]
        return sorted(dirs, key=lambda p: int(p.name.split("-")[-1]))

    def latest_checkpoint(self) -> Optional[Path]:
        dirs = self.checkpoint_dirs()
        return dirs[-1] if dirs else None

    def resolve_resume_path(self, resume_from_checkpoint: Optional[str]) -> Optional[Path]:
        if not resume_from_checkpoint:
            return None
        if resume_from_checkpoint == "latest":
            latest = self.latest_checkpoint()
            if latest is None:
                logger.info("resume_from_checkpoint='latest' requested but no checkpoints found; starting fresh.")
            return latest
        path = Path(resume_from_checkpoint)
        if not path.exists():
            logger.warning("Requested resume checkpoint %s does not exist; starting fresh.", path)
            return None
        return path

    def save(self, accelerator, step: int) -> Path:
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(ckpt_dir))
        logger.info("Saved checkpoint at step %d -> %s", step, ckpt_dir)
        self._prune()
        return ckpt_dir

    def _prune(self) -> None:
        dirs = self.checkpoint_dirs()
        excess = len(dirs) - self.checkpoints_total_limit
        for old_dir in dirs[:max(excess, 0)]:
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.info("Pruned old checkpoint %s (checkpoints_total_limit=%d)", old_dir, self.checkpoints_total_limit)


class ValidationImageCallback:
    """Renders a fixed set of prompts with fixed seeds at every validation
    step and saves them (plus a contact sheet) under
    `output_dir/step_<N>/`, so checkpoints can be compared side by side."""

    def __init__(
        self,
        prompts: list[str],
        num_images: int,
        output_dir: "str | Path",
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
    ):
        self.prompts = prompts
        self.num_images = num_images
        self.output_dir = Path(output_dir)
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height

    def run(self, pipeline, step: int, tb_writer=None, seed: int = 0) -> list[Path]:
        import torch
        from PIL import Image

        step_dir = self.output_dir / f"step_{step:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        saved, images = [], []
        for p_idx, prompt in enumerate(self.prompts):
            for i in range(self.num_images):
                # CPU generator: identical noise on any device, every step.
                generator = torch.Generator(device="cpu").manual_seed(seed + i)
                image = pipeline(
                    prompt,
                    negative_prompt=self.negative_prompt or None,
                    generator=generator,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    width=self.width,
                    height=self.height,
                ).images[0]
                path = step_dir / f"prompt{p_idx}_seed{seed + i}.png"
                image.save(path)
                saved.append(path)
                images.append(image)
                if tb_writer is not None:
                    import numpy as np

                    tb_writer.add_image(f"validation/prompt{p_idx}_{i}", np.asarray(image).transpose(2, 0, 1), global_step=step)

        if images:
            cols = self.num_images
            rows = len(self.prompts)
            sheet = Image.new("RGB", (cols * self.width, rows * self.height), "white")
            for k, img in enumerate(images):
                sheet.paste(img, ((k % cols) * self.width, (k // cols) * self.height))
            sheet.save(step_dir / "grid.jpg", quality=90)
        (step_dir / "prompts.txt").write_text("\n".join(self.prompts) + "\n", encoding="utf-8")
        logger.info("Saved %d validation images at step %d -> %s", len(saved), step, step_dir)
        return saved


class LossLogger:
    """Per-optimizer-step loss/LR history, written to a CSV (always) and to
    TensorBoard when available."""

    def __init__(self, logging_dir: "str | Path", report_to: str = "tensorboard"):
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.logging_dir / "train_loss.csv"
        self.history: list[dict] = []
        self._unflushed: list[dict] = []
        self.writer = None
        if self.csv_path.exists():
            # Resumed run: keep earlier rows.
            import csv

            with self.csv_path.open(encoding="utf-8") as f:
                self.history = [
                    {"step": int(r["step"]), "loss": float(r["loss"]), "lr": float(r["lr"])} for r in csv.DictReader(f)
                ]
        if report_to == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.logging_dir))
            except Exception:
                logger.warning("tensorboard not installed or not usable; logging loss to CSV only.")

    def record(self, step: int, loss: float, lr: float) -> None:
        # Drop rows from a previous run beyond this step (resume rewinds).
        while self.history and self.history[-1]["step"] >= step:
            self.history.pop()
        row = {"step": step, "loss": loss, "lr": lr}
        self.history.append(row)
        self._unflushed.append(row)
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/lr", lr, step)
        if len(self._unflushed) >= 50:
            self.flush()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        logger.info("step=%d %s=%.5f", step, tag, value)

    def recent_mean(self, n: int) -> Optional[float]:
        rows = self.history[-n:]
        return sum(r["loss"] for r in rows) / len(rows) if rows else None

    def flush(self) -> None:
        import csv

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr"])
            writer.writeheader()
            writer.writerows(self.history)
        self._unflushed.clear()
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
