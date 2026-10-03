"""Inference-time generation engine.

Wraps a Diffusers `StableDiffusionPipeline` (optionally
`StableDiffusionControlNetPipeline`), loads the trained Ravi Varma LoRA
weights if present, and exposes a single `generate()` call used by both
`scripts/generate.py` and the Gradio app.

Design choices mandated by the project spec:
  * If no LoRA checkpoint exists yet, we do NOT silently generate with the
    bare base model and pretend it's "Ravi Varma style" -- `RaviVarmaGenerator.
    __init__` records `lora_loaded=False` and every result's metadata carries
    that flag so the UI can show a clear "checkpoint missing" message.
  * The heavy imports (torch/diffusers) happen inside `__init__`, so
    building a `GenerationConfig` and inspecting it doesn't require them.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ravi_varma.config import GenerationConfig
from ravi_varma.generation.controlnet import CONTROLNET_MODEL_IDS, make_conditioning_image
from ravi_varma.utils.device import detect_device, resolve_optimizations
from ravi_varma.utils.logging import get_logger

logger = get_logger(__name__)

SCHEDULER_MAP = {
    "dpmsolver_multistep": "DPMSolverMultistepScheduler",
    "euler_a": "EulerAncestralDiscreteScheduler",
    "ddim": "DDIMScheduler",
    "pndm": "PNDMScheduler",
}


@dataclass
class GenerationResult:
    image: Image.Image
    metadata: dict
    image_path: Optional[Path] = None
    metadata_path: Optional[Path] = None


LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def resolve_lora_weights(lora_path: "str | Path | None") -> Optional[Path]:
    """Accept a final LoRA directory, a `checkpoint-<step>` directory, or a
    `.safetensors` file; return the weights file, or None if there is none.
    A directory that only contains `checkpoint-*` sub-folders (training
    still running / interrupted) resolves to its latest checkpoint."""
    if not lora_path:
        return None
    path = Path(lora_path)
    if path.is_file() and path.suffix == ".safetensors":
        return path
    if path.is_dir():
        if (path / LORA_WEIGHTS_NAME).exists():
            return path / LORA_WEIGHTS_NAME
        checkpoints = sorted(
            (p for p in path.glob("checkpoint-*") if (p / LORA_WEIGHTS_NAME).exists()),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            logger.warning("No final weights in %s; using latest checkpoint %s", path, checkpoints[-1])
            return checkpoints[-1] / LORA_WEIGHTS_NAME
    return None


def _read_training_config(weights_file: Path) -> dict:
    """training_config.yaml written next to the final weights (or in the
    parent of a checkpoint dir), if present."""
    import yaml

    for candidate in (weights_file.parent / "training_config.yaml", weights_file.parent.parent / "training_config.yaml"):
        if candidate.exists():
            try:
                return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}
    return {}


class LoraCheckpointMissingError(RuntimeError):
    """Raised (or, in non-strict mode, only logged) when the LoRA checkpoint
    directory does not exist. Distinguishing this from a generic RuntimeError
    lets the Gradio app show a specific, actionable setup message instead of
    a stack trace."""


def _resolve_scheduler(pipeline, scheduler_key: str):
    import diffusers

    cls_name = SCHEDULER_MAP.get(scheduler_key)
    if cls_name is None:
        raise ValueError(f"Unknown scheduler '{scheduler_key}'.")
    scheduler_cls = getattr(diffusers, cls_name)
    return scheduler_cls.from_config(pipeline.scheduler.config)


class RaviVarmaGenerator:
    def __init__(self, config: GenerationConfig, allow_missing_lora: bool = True, use_lora: bool = True):
        self.config = config
        self.device_info = detect_device(config.hardware.device)
        self.plan = resolve_optimizations(
            self.device_info,
            requested_attention_slicing=config.hardware.attention_slicing,
            requested_vae_slicing=config.hardware.vae_slicing,
            requested_vae_tiling=config.hardware.vae_tiling,
            requested_model_cpu_offload=config.hardware.enable_model_cpu_offload,
        )
        self._pipeline = None
        self._controlnet_pipeline = None
        self.lora_loaded = False
        self.lora_weights: Optional[Path] = None
        self._loaded = False
        self.allow_missing_lora = allow_missing_lora
        # use_lora=False deliberately renders with the base model only (used
        # for base-vs-LoRA comparisons); results are labelled lora_loaded=False.
        self.use_lora = use_lora

    # ------------------------------------------------------------------ #
    def _check_ml_stack(self) -> None:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "torch/diffusers are not usable in this environment "
                f"({e.__class__.__name__}: {e}), so real image generation cannot run here. Install "
                "requirements.txt cleanly on a machine with a GPU (or Colab) to generate images. "
                "See README 'Hardware Requirements'."
            ) from e

    def _dtype(self):
        import torch

        if self.config.hardware.dtype == "float32":
            return torch.float32
        if self.config.hardware.dtype == "float16":
            return torch.float16
        # auto: fp16 on CUDA and Apple MPS (half the memory; an 8 GB Mac swaps
        # heavily in fp32), fp32 on CPU where fp16 kernels are slow/missing.
        # NB: on MPS, fp16 must NOT be combined with attention slicing -- that
        # combination produces NaN latents (black images) with torch 2.4;
        # resolve_optimizations() therefore leaves slicing off on MPS.
        return torch.float16 if self.device_info.device in ("cuda", "mps") else torch.float32

    def load(self) -> None:
        """Lazily load the base pipeline + LoRA weights. Safe to call more
        than once (no-ops after the first successful load)."""
        if self._loaded:
            return
        self._check_ml_stack()

        from diffusers import StableDiffusionPipeline

        cfg = self.config
        dtype = self._dtype()

        logger.info("Loading base model %s on %s (dtype=%s)...", cfg.model.base_model, self.device_info.device, dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model.base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
        pipe.scheduler = _resolve_scheduler(pipe, cfg.model.scheduler)
        if self.plan.model_cpu_offload:
            pipe.enable_model_cpu_offload()  # manages device placement itself; must not be combined with .to()
        else:
            pipe = pipe.to(self.device_info.device)

        if self.plan.attention_slicing:
            pipe.enable_attention_slicing()
        if self.plan.vae_slicing:
            pipe.enable_vae_slicing()
        if self.plan.vae_tiling:
            pipe.enable_vae_tiling()

        weights = resolve_lora_weights(cfg.model.lora_path) if self.use_lora else None
        if weights is not None:
            trained = _read_training_config(weights)
            if trained.get("base_model") and trained["base_model"] != cfg.model.base_model:
                logger.warning(
                    "LoRA was trained on %s but generating with %s; results may be degraded.",
                    trained["base_model"], cfg.model.base_model,
                )
            trained_token = (trained.get("style") or {}).get("trigger_token")
            if trained_token and trained_token != cfg.style.trigger_token:
                logger.warning("LoRA trigger token is %r but config uses %r.", trained_token, cfg.style.trigger_token)
            try:
                pipe.load_lora_weights(str(weights.parent), weight_name=weights.name)
                self.lora_loaded = True
                self.lora_weights = weights
                logger.info("Loaded Ravi Varma LoRA weights from %s", weights)
            except Exception as e:
                logger.error("Found LoRA weights %s but failed to load them: %s", weights, e)
                self.lora_loaded = False
        elif self.use_lora:
            self.lora_loaded = False
            message = (
                f"No trained LoRA weights found at '{cfg.model.lora_path}'. Generation will use the "
                "*base* Stable Diffusion model only -- output will NOT reflect Ravi Varma's "
                "style. Train on a GPU (notebooks/03_lora_training.ipynb) and copy "
                "checkpoints/lora/ravi_varma/ here, or pass --lora-path."
            )
            if self.allow_missing_lora:
                logger.warning(message)
            else:
                raise LoraCheckpointMissingError(message)
        else:
            logger.info("LoRA disabled (use_lora=False): generating with the base model only.")

        self._pipeline = pipe
        self._loaded = True

    def _load_controlnet_pipeline(self, control_type: str):
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        model_id = CONTROLNET_MODEL_IDS[control_type]
        dtype = self._dtype()
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype)
        cn_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.model.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device_info.device)
        cn_pipe.scheduler = self._pipeline.scheduler
        if self.lora_loaded and self.lora_weights is not None:
            cn_pipe.load_lora_weights(str(self.lora_weights.parent), weight_name=self.lora_weights.name)
        if self.plan.attention_slicing:
            cn_pipe.enable_attention_slicing()
        return cn_pipe

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        lora_scale: Optional[float] = None,
        control_image: Optional[Image.Image] = None,
        control_type: Optional[str] = None,
        controlnet_strength: Optional[float] = None,
        save: bool = True,
        output_dir: "str | Path | None" = None,
        name_prefix: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> GenerationResult:
        self.load()
        import torch

        cfg = self.config
        gen_cfg = cfg.generation
        steps = steps or gen_cfg.steps
        guidance_scale = guidance_scale if guidance_scale is not None else gen_cfg.guidance_scale
        width = width or gen_cfg.width
        height = height or gen_cfg.height
        lora_scale = lora_scale if lora_scale is not None else gen_cfg.lora_scale
        negative_prompt = negative_prompt if negative_prompt is not None else cfg.negative_prompt

        used_seed = seed if seed is not None else int.from_bytes(__import__("os").urandom(4), "little")
        # CPU generator: the initial noise (and therefore the image) for a
        # given seed is identical on CUDA, MPS and CPU.
        generator = torch.Generator(device="cpu").manual_seed(used_seed)

        cross_attention_kwargs = {"scale": lora_scale} if self.lora_loaded else None

        use_controlnet = control_image is not None and (control_type or cfg.controlnet.type)
        if use_controlnet:
            resolved_type = control_type or cfg.controlnet.type
            conditioning = make_conditioning_image(
                control_image,
                control_type=resolved_type,
                canny_low_threshold=cfg.controlnet.canny_low_threshold,
                canny_high_threshold=cfg.controlnet.canny_high_threshold,
            )
            if self._controlnet_pipeline is None:
                self._controlnet_pipeline = self._load_controlnet_pipeline(resolved_type)
            strength = controlnet_strength if controlnet_strength is not None else cfg.controlnet.strength
            result = self._controlnet_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioning,
                controlnet_conditioning_scale=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        image = result.images[0]
        if self.device_info.device == "mps":
            # Release MPS cached blocks between images; on 8 GB unified memory
            # the cache otherwise grows until the process is swapped out.
            torch.mps.empty_cache()
        if image.getextrema() == ((0, 0), (0, 0), (0, 0)):
            # NaNs from fp16 overflow decode to pure black; never save that silently.
            raise RuntimeError(
                "Generation produced an all-black image (numerical overflow, usually fp16 on this "
                "device). Set hardware.dtype: float32 in configs/generation.yaml."
            )
        metadata = {
            **(extra_metadata or {}),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": used_seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "lora_scale": lora_scale,
            "lora_loaded": self.lora_loaded,
            "lora_weights": str(self.lora_weights) if self.lora_loaded else None,
            "base_model": cfg.model.base_model,
            "scheduler": cfg.model.scheduler,
            "controlnet_enabled": bool(use_controlnet),
            "controlnet_type": (control_type or cfg.controlnet.type) if use_controlnet else None,
            "controlnet_strength": controlnet_strength if use_controlnet else None,
            "device": self.device_info.device,
            "dtype": str(self._dtype()).replace("torch.", ""),
            "timestamp": time.time(),
            "disclaimer": "Ravi Varma-inspired generative artwork. AI-generated; not an authentic "
            "Raja Ravi Varma painting. Evaluation scores (if present) measure statistical "
            "similarity, not artistic authenticity.",
        }

        image_path = metadata_path = None
        if save:
            image_path, metadata_path = self._save(image, metadata, output_dir=output_dir, name_prefix=name_prefix)

        return GenerationResult(image=image, metadata=metadata, image_path=image_path, metadata_path=metadata_path)

    def _save(
        self, image: Image.Image, metadata: dict, output_dir: "str | Path | None" = None, name_prefix: Optional[str] = None
    ) -> tuple[Path, Optional[Path]]:
        out_dir = Path(output_dir or self.config.output.generated_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{name_prefix}_seed{metadata['seed']}" if name_prefix else f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        image_path = out_dir / f"{stamp}.png"
        image.save(image_path)
        metadata_path = None
        if self.config.output.save_metadata:
            metadata_path = out_dir / f"{stamp}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Saved generated image -> %s", image_path)
        return image_path, metadata_path
