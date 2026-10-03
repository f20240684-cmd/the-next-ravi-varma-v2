"""LoRA fine-tuning of a Stable Diffusion 1.5 UNet using Diffusers + PEFT +
Accelerate.

This mirrors the structure of Diffusers' official `train_text_to_image_lora`
example, wrapped in a class so it can be driven both from
`scripts/train_lora.py` and from the Colab notebook, and adapted to:
  * load its own dataset (`RaviVarmaImageCaptionDataset`) built from our
    JSONL manifest, with aspect-ratio buckets (each painting is trained at
    its prepared size, e.g. 448x640, instead of a square crop),
  * inject the configurable style trigger token (it is part of every
    caption; the text encoder stays frozen),
  * respect the shared `MemoryOptimizationPlan` from `utils/device.py`,
  * checkpoint/resume/validate via `training/callbacks.py`, exporting a
    loadable LoRA file with every checkpoint so checkpoints can be compared.

Memory layout follows the Diffusers reference: frozen VAE / text encoder /
UNet weights are cast to the mixed-precision dtype (fp16 on a T4) and only
the LoRA parameters are kept in fp32, so the optimizer and gradient scaler
operate on full-precision weights.

All heavy imports (torch, diffusers, peft, accelerate, bitsandbytes,
xformers) are performed lazily inside methods, never at module import time,
so `from ravi_varma.training.lora_trainer import LoraTrainer` succeeds even
on a machine without the ML stack installed (e.g. to unit-test config
handling). Actually calling `.train()` without those packages installed
raises a clear `RuntimeError` rather than failing deep in the call stack.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Optional

from ravi_varma.config import LoraTrainingConfig
from ravi_varma.data.dataset import BucketBatchSampler, RaviVarmaImageCaptionDataset, load_manifest
from ravi_varma.training.callbacks import CheckpointManager, LossLogger, ValidationImageCallback
from ravi_varma.utils.device import detect_device, resolve_optimizations
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_PACKAGES = ("torch", "diffusers", "transformers", "peft", "accelerate")
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def check_ml_stack_available() -> Optional[str]:
    """Return None if everything needed for real training is importable,
    otherwise a human-readable message naming what's missing (or broken --
    e.g. a partial/corrupted install missing shared libraries)."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except Exception as e:  # broad: broken installs raise OSError/RuntimeError, not just ImportError
            missing.append(f"{pkg} ({e.__class__.__name__}: {e})")
    if missing:
        return (
            "Cannot run LoRA training: missing or broken packages "
            f"{missing}. Install requirements.txt cleanly on a GPU machine or Colab (see "
            "notebooks/03_lora_training.ipynb)."
        )
    return None


class CaptionCollator:
    """Batch collation as a top-level class (not a closure) so DataLoader
    worker processes can pickle it on spawn-based platforms."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict]) -> dict:
        import torch

        pixel_values = torch.stack([e["pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer(
            [e["text"] for e in examples],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}


class LoraTrainer:
    def __init__(self, config: LoraTrainingConfig):
        self.config = config
        self.device_info = detect_device(preferred="auto")
        self.plan = resolve_optimizations(
            self.device_info,
            requested_mixed_precision=config.training.mixed_precision,
            requested_xformers=config.training.enable_xformers,
        )
        self._accelerator = None
        self._unet = None
        self._text_encoder = None
        self._vae = None
        self._tokenizer = None
        self._noise_scheduler = None
        self._pipeline_cls = None
        self._weight_dtype = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _load_models(self):
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        cfg = self.config
        kwargs = {"revision": cfg.revision}
        if cfg.variant:
            kwargs["variant"] = cfg.variant

        if cfg.model_family == "sdxl":
            # SDXL has two text encoders/tokenizers; kept as a documented
            # extension point -- the supported path is SD1.5.
            raise NotImplementedError(
                "SDXL training path is scaffolded via configs/lora_sdxl.yaml but requires "
                ">= 16GB VRAM and the dual text-encoder handling in Diffusers' "
                "train_text_to_image_lora_sdxl.py. Contributions welcome -- see README."
            )

        self._tokenizer = CLIPTokenizer.from_pretrained(cfg.base_model, subfolder="tokenizer", **kwargs)
        self._text_encoder = CLIPTextModel.from_pretrained(cfg.base_model, subfolder="text_encoder", **kwargs)
        self._vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae", **kwargs)
        self._unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet", **kwargs)
        self._noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
        self._pipeline_cls = StableDiffusionPipeline

    def _apply_lora(self, device) -> list:
        """Freeze the base model, move frozen weights to the mixed-precision
        dtype, attach LoRA adapters and return the (fp32) trainable params."""
        import torch
        from diffusers.training_utils import cast_training_params
        from peft import LoraConfig as PeftLoraConfig

        lora_cfg = self.config.lora
        self._vae.requires_grad_(False)
        self._text_encoder.requires_grad_(False)
        self._unet.requires_grad_(False)

        self._unet.to(device, dtype=self._weight_dtype)
        self._vae.to(device, dtype=self._weight_dtype)
        self._text_encoder.to(device, dtype=self._weight_dtype)

        self._unet.add_adapter(
            PeftLoraConfig(
                r=lora_cfg.rank,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                init_lora_weights="gaussian",
                target_modules=lora_cfg.target_modules,
            )
        )
        if lora_cfg.train_text_encoder:
            self._text_encoder.add_adapter(
                PeftLoraConfig(
                    r=lora_cfg.rank,
                    lora_alpha=lora_cfg.alpha,
                    lora_dropout=lora_cfg.dropout,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
            )

        # Only the adapters are trained, and in fp32 (the frozen base stays
        # in fp16/bf16). Upcasting is what makes fp16 AMP + GradScaler work.
        models = [self._unet] + ([self._text_encoder] if lora_cfg.train_text_encoder else [])
        if self._weight_dtype != torch.float32:
            cast_training_params(models, dtype=torch.float32)

        trainable = [(n, p) for m in models for n, p in m.named_parameters() if p.requires_grad]
        non_lora = [n for n, _ in trainable if "lora" not in n]
        if non_lora:
            raise RuntimeError(f"Non-LoRA parameters are trainable, base model is not frozen: {non_lora[:5]}")
        return [p for _, p in trainable]

    def _build_optimizer(self, trainable_params):
        opt_cfg = self.config.optimizer
        optimizer_cls = None
        if opt_cfg.use_8bit_adam:
            if self.device_info.device != "cuda":
                logger.warning("use_8bit_adam=True requires CUDA (bitsandbytes); using standard AdamW.")
            else:
                try:
                    import bitsandbytes as bnb

                    optimizer_cls = bnb.optim.AdamW8bit
                except Exception as e:  # missing, or installed without a matching CUDA/Triton build
                    logger.warning("use_8bit_adam=True but bitsandbytes is unusable (%s); using standard AdamW.", e)
        if optimizer_cls is None:
            import torch

            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_params,
            lr=opt_cfg.learning_rate,
            betas=(opt_cfg.adam_beta1, opt_cfg.adam_beta2),
            weight_decay=opt_cfg.adam_weight_decay,
            eps=opt_cfg.adam_epsilon,
        )

    def print_startup_summary(self, dataset, num_batches: int, max_train_steps: int, trainable_params: list, total_params: int) -> None:
        cfg = self.config
        trainable_count = sum(p.numel() for p in trainable_params)
        pct = 100 * trainable_count / max(total_params, 1)
        effective_batch = cfg.training.train_batch_size * cfg.training.gradient_accumulation_steps
        sizes: dict[str, int] = {}
        for i in range(len(dataset)):
            w, h = dataset.size_of(i)
            sizes[f"{w}x{h}"] = sizes.get(f"{w}x{h}", 0) + 1
        logger.info("=== The Next Ravi Varma -- LoRA Training ===")
        logger.info("Base model: %s", cfg.base_model)
        logger.info("Device: %s (%s)", self.device_info.device, self.device_info.gpu_name or "n/a")
        if self.device_info.total_vram_gb:
            logger.info("VRAM: %.1f GB", self.device_info.total_vram_gb)
        logger.info("Dataset: %d samples, resolutions %s", len(dataset), sizes)
        logger.info(
            "Optimizer steps: %d (batch %d x grad-accum %d = effective batch %d; ~%.1f epochs)",
            max_train_steps, cfg.training.train_batch_size, cfg.training.gradient_accumulation_steps,
            effective_batch, max_train_steps * cfg.training.gradient_accumulation_steps / max(num_batches, 1),
        )
        logger.info(
            "LoRA: rank %d, alpha %d, dropout %.2f, targets %s",
            cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target_modules,
        )
        logger.info("Trainable parameters: %d (%.3f%% of %d UNet params)", trainable_count, pct, total_params)
        logger.info("Mixed precision: %s | frozen weights: %s | xFormers: %s", self.plan.mixed_precision, self._weight_dtype, self.plan.use_xformers)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self) -> Path:
        missing = check_ml_stack_available()
        if missing:
            raise RuntimeError(missing)

        import torch
        import torch.nn.functional as F
        from accelerate import Accelerator
        from accelerate.utils import set_seed

        cfg = self.config
        set_seed(cfg.training.seed)

        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            mixed_precision=self.plan.mixed_precision,
            log_with=None,  # LossLogger owns TensorBoard + CSV logging
            project_dir=cfg.logging.logging_dir,
        )
        self._accelerator = accelerator
        self._weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.plan.mixed_precision, torch.float32)

        self._load_models()
        trainable_params = self._apply_lora(accelerator.device)

        if cfg.training.gradient_checkpointing:
            self._unet.enable_gradient_checkpointing()
        if self.plan.use_xformers:
            try:
                self._unet.enable_xformers_memory_efficient_attention()
            except Exception as e:  # pragma: no cover - hardware dependent
                logger.warning("Could not enable xFormers attention: %s", e)

        manifest = load_manifest(cfg.dataset.manifest_file)
        if not manifest:
            raise RuntimeError(f"Manifest {cfg.dataset.manifest_file} is empty. Run scripts/prepare_dataset.py first.")
        dataset = RaviVarmaImageCaptionDataset(
            manifest,
            resolution=cfg.dataset.resolution,
            center_crop=cfg.dataset.center_crop,
            random_flip=cfg.dataset.random_flip,
            keep_aspect=cfg.dataset.aspect_bucketing,
        )
        num_workers = cfg.training.dataloader_num_workers if self.device_info.device == "cuda" else 0
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(dataset, cfg.training.train_batch_size, seed=cfg.training.seed),
            collate_fn=CaptionCollator(self._tokenizer),
            num_workers=num_workers,
        )

        optimizer = self._build_optimizer(trainable_params)

        steps_per_epoch = math.ceil(len(dataloader) / cfg.training.gradient_accumulation_steps)
        max_train_steps = cfg.training.max_train_steps
        if cfg.training.num_train_epochs:
            max_train_steps = steps_per_epoch * cfg.training.num_train_epochs

        from diffusers.optimization import get_scheduler

        # Accelerate's prepared scheduler only advances on real optimizer
        # steps (after gradient accumulation), so warmup/total are expressed
        # in optimizer steps -- scaled by process count, as in the Diffusers
        # reference script. (Earlier versions multiplied by grad-accum,
        # silently stretching the 100-step warmup to 400 steps.)
        lr_scheduler = get_scheduler(
            cfg.optimizer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.optimizer.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=max_train_steps * accelerator.num_processes,
        )

        self._unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            self._unet, optimizer, dataloader, lr_scheduler
        )

        total_params = sum(p.numel() for p in self._unet.parameters())
        self.print_startup_summary(dataset, len(dataloader), max_train_steps, trainable_params, total_params)

        output_dir = Path(cfg.checkpointing.output_dir)
        checkpoint_manager = CheckpointManager(output_dir, cfg.checkpointing.checkpoints_total_limit)
        loss_logger = LossLogger(cfg.logging.logging_dir, cfg.logging.report_to)
        validator = ValidationImageCallback(
            prompts=cfg.validation.prompts(),
            num_images=cfg.validation.num_validation_images,
            output_dir=Path(cfg.logging.logging_dir) / "validation_images",
            negative_prompt=cfg.validation.negative_prompt,
            num_inference_steps=cfg.validation.num_inference_steps,
            guidance_scale=cfg.validation.guidance_scale,
            width=cfg.validation.width,
            height=cfg.validation.height,
        )

        global_step = 0
        resume_path = checkpoint_manager.resolve_resume_path(cfg.checkpointing.resume_from_checkpoint)
        if resume_path is not None:
            accelerator.load_state(str(resume_path))
            global_step = int(str(resume_path).split("-")[-1])
            # The data order restarts from a fresh shuffled epoch; optimizer,
            # LR scheduler, gradient scaler and RNG states are restored.
            logger.info("Resumed from checkpoint %s at step %d", resume_path, global_step)
        elif cfg.validation.run_at_start and accelerator.is_main_process:
            self._run_validation(accelerator, validator, 0, loss_logger)

        progress_start = time.time()
        self._unet.train()
        accum_loss, accum_count, nonfinite = 0.0, 0, 0
        done = global_step >= max_train_steps
        while not done:
            for batch in dataloader:
                with accelerator.accumulate(self._unet):
                    with torch.no_grad():
                        latents = self._vae.encode(batch["pixel_values"].to(dtype=self._weight_dtype)).latent_dist.sample()
                        latents = latents * self._vae.config.scaling_factor
                        encoder_hidden_states = self._text_encoder(batch["input_ids"].to(accelerator.device))[0]

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, self._noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()
                    noisy_latents = self._noise_scheduler.add_noise(latents, noise, timesteps)

                    model_pred = self._unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self._noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self._noise_scheduler.config.prediction_type == "v_prediction":
                        target = self._noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type {self._noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    if not torch.isfinite(loss):
                        nonfinite += 1
                        if nonfinite > 10:
                            raise RuntimeError("Loss was NaN/inf more than 10 times; aborting (try mixed_precision: no).")
                        logger.warning("Non-finite loss at step %d; skipping this batch.", global_step)
                        optimizer.zero_grad()
                        continue

                    accum_loss += loss.detach().item()
                    accum_count += 1
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if not accelerator.sync_gradients:
                    continue

                global_step += 1
                step_loss = accum_loss / max(accum_count, 1)  # mean over the accumulated micro-batches
                accum_loss, accum_count = 0.0, 0
                loss_logger.record(global_step, step_loss, lr_scheduler.get_last_lr()[0])

                if global_step % cfg.logging.log_every_n_steps == 0:
                    elapsed = time.time() - progress_start
                    logger.info(
                        "step %d/%d | loss %.4f (avg last %d: %.4f) | lr %.2e | %.0fs elapsed",
                        global_step, max_train_steps, step_loss, cfg.logging.log_every_n_steps,
                        loss_logger.recent_mean(cfg.logging.log_every_n_steps), lr_scheduler.get_last_lr()[0], elapsed,
                    )

                if accelerator.is_main_process and global_step % cfg.checkpointing.checkpointing_steps == 0:
                    ckpt_dir = checkpoint_manager.save(accelerator, global_step)
                    self._save_lora_weights(accelerator, ckpt_dir)
                    loss_logger.flush()

                if (
                    accelerator.is_main_process
                    and cfg.validation.validation_steps
                    and global_step % cfg.validation.validation_steps == 0
                ):
                    self._run_validation(accelerator, validator, global_step, loss_logger)

                if global_step >= max_train_steps:
                    done = True
                    break

        accelerator.wait_for_everyone()
        final_path = self._save_final(accelerator, cfg, dataset, loss_logger, time.time() - progress_start)
        loss_logger.close()
        return final_path

    def _run_validation(self, accelerator, validator: ValidationImageCallback, step: int, loss_logger: LossLogger) -> None:
        """Render the fixed validation prompts with the current adapter,
        re-using the already-loaded modules (no reload from disk)."""
        import torch
        from diffusers import DPMSolverMultistepScheduler

        unet = accelerator.unwrap_model(self._unet)
        try:
            pipeline = self._pipeline_cls(
                vae=self._vae,
                text_encoder=self._text_encoder,
                tokenizer=self._tokenizer,
                unet=unet,
                scheduler=DPMSolverMultistepScheduler.from_config(self._noise_scheduler.config),
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
            pipeline.set_progress_bar_config(disable=True)
            unet.eval()
            autocast = (
                torch.autocast(accelerator.device.type, dtype=self._weight_dtype)
                if accelerator.device.type == "cuda" and self._weight_dtype != torch.float32
                else contextlib.nullcontext()
            )
            with torch.no_grad(), autocast:
                validator.run(pipeline, step, tb_writer=loss_logger.writer, seed=self.config.training.seed)
        except Exception as e:  # pragma: no cover - hardware dependent; never kill a long run over a preview
            logger.warning("Validation image generation failed at step %d: %s", step, e)
        finally:
            unet.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_lora_weights(self, accelerator, directory: Path) -> Path:
        """Export the current UNet adapter in the Diffusers LoRA format
        (`pytorch_lora_weights.safetensors`), loadable with
        `pipeline.load_lora_weights(directory)`."""
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        unwrapped_unet = accelerator.unwrap_model(self._unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        text_encoder_layers = None
        if self.config.lora.train_text_encoder:
            text_encoder_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(self._text_encoder))
        self._pipeline_cls.save_lora_weights(
            save_directory=str(directory),
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_layers,
            safe_serialization=True,
        )
        return Path(directory) / LORA_WEIGHTS_NAME

    def _save_final(self, accelerator, cfg: LoraTrainingConfig, dataset, loss_logger: LossLogger, train_seconds: float) -> Path:
        import yaml

        output_dir = Path(cfg.checkpointing.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = self._save_lora_weights(accelerator, output_dir)

        # Persist the exact config + a run summary next to the weights, so
        # `generate.py` can check base_model / trigger_token compatibility
        # and the checkpoint is self-describing when shared.
        (output_dir / "training_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
        history = loss_logger.history
        summary = {
            "base_model": cfg.base_model,
            "trigger_token": cfg.style.trigger_token,
            "lora": cfg.lora.model_dump(),
            "dataset_manifest": cfg.dataset.manifest_file,
            "num_training_samples": len(dataset),
            "optimizer_steps": history[-1]["step"] if history else 0,
            "train_seconds_this_session": round(train_seconds, 1),
            "device": self.device_info.device,
            "gpu": self.device_info.gpu_name,
            "mixed_precision": self.plan.mixed_precision,
            "final_loss_mean_last_50": loss_logger.recent_mean(50),
            "loss_history_csv": str(loss_logger.csv_path),
            "weights_file": str(weights),
            "checkpoints": [str(p) for p in CheckpointManager(output_dir, cfg.checkpointing.checkpoints_total_limit).checkpoint_dirs()],
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        loss_logger.flush()
        if loss_logger.csv_path and Path(loss_logger.csv_path).exists():
            (output_dir / "loss_history.csv").write_text(Path(loss_logger.csv_path).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Saved final LoRA weights -> %s", weights)
        return output_dir
