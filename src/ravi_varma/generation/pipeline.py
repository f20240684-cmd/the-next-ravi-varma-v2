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
    control_image: Optional[Image.Image] = None  # the pose/edge map ControlNet followed, if any


LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h), largest first, from OpenCV's frontal-face Haar cascade.
    shortcut: Haar misses profiles and very stylised faces; swap in a DNN detector if that matters."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    min_side = max(24, min(image.size) // 25)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
    return sorted((tuple(int(v) for v in f) for f in found), key=lambda b: -b[2] * b[3])


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
        self._controlnet_pipelines: dict = {}
        self._img2img = None
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
        if cfg.model.vae:
            from diffusers import AutoencoderKL

            # Same latent space as SD1.5's VAE, fine-tuned decoder: cleaner faces/eyes, no retraining needed.
            pipe.vae = AutoencoderKL.from_pretrained(cfg.model.vae, torch_dtype=dtype)
            logger.info("Using VAE %s", cfg.model.vae)
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
        if not self.plan.model_cpu_offload:
            controlnet = controlnet.to(self.device_info.device)
        # Reuse the loaded UNet/VAE/text encoder (LoRA already injected) instead of loading a
        # second copy of SD1.5: halves memory, which is what lets this run on an 8 GB Mac.
        components = {k: v for k, v in self._pipeline.components.items() if k != "controlnet"}
        cn_pipe = StableDiffusionControlNetPipeline(**components, controlnet=controlnet)
        if self.plan.model_cpu_offload:
            cn_pipe.enable_model_cpu_offload()
        return cn_pipe

    def _redraw(self, image: Image.Image, strength: float, prompt: str, negative_prompt: str, steps: int,
                guidance_scale: float, seed: int, cross_attention_kwargs) -> Image.Image:
        """img2img pass with the already-loaded UNet/VAE/text encoder (LoRA included)."""
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline

        if self._img2img is None:
            components = {k: v for k, v in self._pipeline.components.items() if k != "controlnet"}
            self._img2img = StableDiffusionImg2ImgPipeline(**components)
        out = self._img2img(
            prompt=prompt, negative_prompt=negative_prompt, image=image, strength=strength,
            num_inference_steps=steps, guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed), cross_attention_kwargs=cross_attention_kwargs,
        ).images[0]
        if self.device_info.device == "mps":
            torch.mps.empty_cache()
        return out

    def _fix_faces(self, image: Image.Image, max_faces: int = 4, **refine) -> int:
        """ADetailer-style: re-draw each face at 512 px, where SD1.5 has room for eyes and lips,
        and blend it back with a feathered oval mask. Modifies `image` in place; returns the count."""
        from PIL import ImageDraw, ImageFilter

        boxes = detect_faces(image)[:max_faces]
        for x, y, w, h in boxes:
            pad = int(max(w, h) * 0.6)  # include hair and chin so the face blends in
            side = max(w, h) + 2 * pad
            cx, cy = x + w // 2, y + h // 2
            box = (max(cx - side // 2, 0), max(cy - side // 2, 0), min(cx + side // 2, image.width), min(cy + side // 2, image.height))
            crop = image.crop(box)
            face_prompt = f"detailed face, clear eyes, {refine['prompt']}"
            fixed = self._redraw(crop.resize((512, 512), Image.LANCZOS), strength=0.4, **{**refine, "prompt": face_prompt})
            fixed = fixed.resize(crop.size, Image.LANCZOS)
            mask = Image.new("L", crop.size, 0)
            inset = int(min(crop.size) * 0.12)
            ImageDraw.Draw(mask).ellipse((inset, inset, crop.width - inset, crop.height - inset), fill=255)
            image.paste(fixed, box[:2], mask.filter(ImageFilter.GaussianBlur(inset / 2)))
        return len(boxes)

    def set_lora(self, lora_path: "str | None") -> None:
        """Switch to another LoRA checkpoint (e.g. checkpoint-500) without reloading SD1.5."""
        self.load()
        if self.lora_loaded:
            self._pipeline.unload_lora_weights()
            self.lora_loaded, self.lora_weights = False, None
        self.config.model.lora_path = lora_path
        weights = resolve_lora_weights(lora_path) if lora_path else None
        if weights is not None:
            self._pipeline.load_lora_weights(str(weights.parent), weight_name=weights.name)
            self.lora_loaded, self.lora_weights = True, weights
            logger.info("Switched LoRA to %s", weights)

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
        face_fix: Optional[bool] = None,
        hires_scale: Optional[float] = None,
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
        conditioning = None
        if use_controlnet:
            from PIL import ImageOps

            resolved_type = control_type or cfg.controlnet.type
            # Crop the reference to the output aspect first, so the pose is not stretched.
            reference = ImageOps.fit(control_image.convert("RGB"), (width, height), method=Image.LANCZOS)
            conditioning = make_conditioning_image(
                reference,
                control_type=resolved_type,
                canny_low_threshold=cfg.controlnet.canny_low_threshold,
                canny_high_threshold=cfg.controlnet.canny_high_threshold,
            ).resize((width, height), Image.LANCZOS)
            if resolved_type not in self._controlnet_pipelines:
                self._controlnet_pipelines[resolved_type] = self._load_controlnet_pipeline(resolved_type)
            strength = controlnet_strength if controlnet_strength is not None else cfg.controlnet.strength
            result = self._controlnet_pipelines[resolved_type](
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
        face_fix = gen_cfg.face_fix if face_fix is None else face_fix
        hires_scale = gen_cfg.hires_scale if hires_scale is None else hires_scale
        refine = dict(prompt=prompt, negative_prompt=negative_prompt, steps=steps, guidance_scale=guidance_scale,
                      seed=used_seed, cross_attention_kwargs=cross_attention_kwargs)
        if hires_scale and hires_scale > 1:
            size = (int(image.width * hires_scale) // 8 * 8, int(image.height * hires_scale) // 8 * 8)
            image = self._redraw(image.resize(size, Image.LANCZOS), strength=0.4, **refine)
        faces_fixed = self._fix_faces(image, **refine) if face_fix else 0
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
            "hires_scale": hires_scale if hires_scale and hires_scale > 1 else None,
            "faces_fixed": faces_fixed if face_fix else None,
            "lora_loaded": self.lora_loaded,
            "lora_weights": str(self.lora_weights) if self.lora_loaded else None,
            "base_model": cfg.model.base_model,
            "scheduler": cfg.model.scheduler,
            "vae": cfg.model.vae,
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
            if conditioning is not None:
                conditioning.save(image_path.with_name(f"{image_path.stem}_control.png"))

        return GenerationResult(
            image=image, metadata=metadata, image_path=image_path, metadata_path=metadata_path, control_image=conditioning
        )

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
