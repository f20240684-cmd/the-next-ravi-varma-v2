"""Centralized, typed configuration loading.

All YAML files under `configs/` are parsed into pydantic models here, so the
rest of the codebase never touches raw dicts or scatters magic numbers.
Every loader accepts either a path to a YAML file or an already-parsed dict
(the latter is what the test suite uses to avoid touching the filesystem).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PathLike = Union[str, Path]


def _load_yaml(path_or_dict: Union[PathLike, dict]) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    path = Path(path_or_dict)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping.")
    return data


# --------------------------------------------------------------------------- #
# Dataset config
# --------------------------------------------------------------------------- #
class ValidationConfig(BaseModel):
    min_resolution: int = 384
    allowed_formats: list[str] = Field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    max_file_size_mb: float = 25
    detect_duplicates: bool = True
    near_duplicate_threshold: int = 40
    require_caption: bool = True


class CurationConfig(BaseModel):
    """Human curation decisions (see configs/dataset.yaml). `exclude` maps a
    raw image path to the reason it is left out of training;
    `duplicate_groups` lists files that reproduce the same artwork."""

    exclude: dict[str, str] = Field(default_factory=dict)
    duplicate_groups: list[list[str]] = Field(default_factory=list)
    holdout: list[str] = Field(default_factory=list)  # accepted images reserved for evaluation
    caption_notes: dict[str, str] = Field(default_factory=dict)  # appended to the training caption


class PreprocessingConfig(BaseModel):
    target_resolution: int = 512
    aspect_bucketing: bool = True
    resize_mode: Literal["center_crop", "pad", "stretch"] = "center_crop"
    interpolation: Literal["lanczos", "bicubic", "bilinear", "nearest"] = "lanczos"
    trim_borders: bool = True


class StyleConfig(BaseModel):
    artist_name: str = "Raja Ravi Varma"
    trigger_token: str = "<rvvarma>"
    caption_prefix_template: str = "{trigger_token}, {caption}"


class DatasetConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    captions_dir: str = "data/captions"
    metadata_dir: str = "data/metadata"
    metadata_file: str = "data/metadata/metadata.jsonl"
    manifest_file: str = "data/processed/manifest.jsonl"
    reference_dir: str = "data/reference"
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)

    @classmethod
    def load(cls, path_or_dict: Union[PathLike, dict] = "configs/dataset.yaml") -> "DatasetConfig":
        return cls(**_load_yaml(path_or_dict))


# --------------------------------------------------------------------------- #
# LoRA training config
# --------------------------------------------------------------------------- #
class TrainingDatasetConfig(BaseModel):
    manifest_file: str = "data/processed/manifest.jsonl"
    resolution: int = 512
    aspect_bucketing: bool = True  # train each image at its prepared bucket size (e.g. 448x640)
    center_crop: bool = True       # only used when aspect_bucketing is false
    random_flip: bool = True


class LoraConfig(BaseModel):
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )
    train_text_encoder: bool = False


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = True


class TrainingLoopConfig(BaseModel):
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1500
    num_train_epochs: Optional[int] = None
    mixed_precision: Literal["fp16", "bf16", "no"] = "fp16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    enable_xformers: bool = True
    dataloader_num_workers: int = 2


class CheckpointingConfig(BaseModel):
    output_dir: str = "checkpoints/lora/ravi_varma"
    checkpointing_steps: int = 250
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = "latest"


class LoggingConfig(BaseModel):
    logging_dir: str = "outputs/logs"
    report_to: Literal["tensorboard", "none"] = "tensorboard"
    log_every_n_steps: int = 10


class TrainingValidationConfig(BaseModel):
    # Fixed prompts + fixed seeds, re-rendered at every validation step, so
    # images from different checkpoints are directly comparable.
    validation_prompts: list[str] = Field(
        default_factory=lambda: ["<rvvarma>, portrait of a royal Indian woman in traditional attire"]
    )
    validation_prompt: Optional[str] = None  # legacy single-prompt field; prepended to validation_prompts if set
    negative_prompt: str = ""
    num_validation_images: int = 1  # images per prompt
    validation_steps: int = 250
    run_at_start: bool = True  # render the same prompts with the untrained adapter (step 0 baseline)
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512

    def prompts(self) -> list[str]:
        prompts = list(self.validation_prompts)
        if self.validation_prompt and self.validation_prompt not in prompts:
            prompts.insert(0, self.validation_prompt)
        return prompts


class LoraTrainingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow `model_*` field names

    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_family: Literal["sd15", "sdxl"] = "sd15"
    revision: Optional[str] = None
    variant: Optional[str] = None

    dataset: TrainingDatasetConfig = Field(default_factory=TrainingDatasetConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingLoopConfig = Field(default_factory=TrainingLoopConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)

    @classmethod
    def load(cls, path_or_dict: Union[PathLike, dict]) -> "LoraTrainingConfig":
        return cls(**_load_yaml(path_or_dict))


# --------------------------------------------------------------------------- #
# Generation / inference config
# --------------------------------------------------------------------------- #
class GenerationModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow `model_*` field names

    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_family: Literal["sd15", "sdxl"] = "sd15"
    lora_path: Optional[str] = "checkpoints/lora/ravi_varma"
    scheduler: Literal["dpmsolver_multistep", "euler_a", "ddim", "pndm"] = "dpmsolver_multistep"


class GenerationParamsConfig(BaseModel):
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    lora_scale: float = 1.0
    seed: Optional[int] = None
    num_images: int = 1

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("width/height must be a multiple of 8 for Stable Diffusion.")
        return v


class ControlNetConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow `model_*` field names

    enabled: bool = False
    type: Literal["openpose", "canny"] = "openpose"
    model_id_openpose: str = "lllyasviel/control_v11p_sd15_openpose"
    model_id_canny: str = "lllyasviel/control_v11p_sd15_canny"
    strength: float = 0.8
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200


class GenerationStyleConfig(BaseModel):
    trigger_token: str = "<rvvarma>"
    style_strength: float = 1.0


class HardwareConfig(BaseModel):
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    dtype: Literal["auto", "float16", "float32"] = "auto"
    attention_slicing: Union[str, bool] = "auto"
    vae_slicing: bool = True
    vae_tiling: bool = False
    enable_model_cpu_offload: Union[str, bool] = "auto"


class OutputConfig(BaseModel):
    generated_dir: str = "outputs/generated"
    save_metadata: bool = True


class GenerationConfig(BaseModel):
    model: GenerationModelConfig = Field(default_factory=GenerationModelConfig)
    generation: GenerationParamsConfig = Field(default_factory=GenerationParamsConfig)
    negative_prompt: str = ""
    controlnet: ControlNetConfig = Field(default_factory=ControlNetConfig)
    style: GenerationStyleConfig = Field(default_factory=GenerationStyleConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path_or_dict: Union[PathLike, dict] = "configs/generation.yaml") -> "GenerationConfig":
        return cls(**_load_yaml(path_or_dict))


def load_config(kind: Literal["dataset", "lora", "generation"], path_or_dict: Union[PathLike, dict, None] = None) -> Any:
    """Convenience dispatcher used by CLI scripts."""
    if kind == "dataset":
        return DatasetConfig.load(path_or_dict or "configs/dataset.yaml")
    if kind == "lora":
        if path_or_dict is None:
            raise ValueError("A LoRA training config path is required.")
        return LoraTrainingConfig.load(path_or_dict)
    if kind == "generation":
        return GenerationConfig.load(path_or_dict or "configs/generation.yaml")
    raise ValueError(f"Unknown config kind: {kind}")
