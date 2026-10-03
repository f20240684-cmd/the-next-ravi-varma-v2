"""End-to-end check of the LoRA trainer on Hugging Face's tiny test pipeline
(`hf-internal-testing/tiny-stable-diffusion-pipe`, ~1 MB, CPU, seconds).

This is NOT a training run of the real model: it exercises the actual code
path -- freezing, LoRA attach, fp32 trainable params, gradient accumulation,
LR scheduling, checkpoint + LoRA export, validation rendering, resume -- so
regressions are caught before spending GPU time. Skipped unless the ML stack
is installed and the tiny model is already in the local HF cache (download
it once with `huggingface-cli download hf-internal-testing/tiny-stable-diffusion-pipe`).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"


def _tiny_model_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(TINY_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _tiny_model_cached(), reason=f"{TINY_MODEL} not in local HF cache")


@pytest.fixture
def cpu_only(monkeypatch):
    """Force CPU so the test behaves the same on CUDA, MPS and CPU hosts."""
    from ravi_varma.training import lora_trainer
    from ravi_varma.utils.device import DeviceInfo

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(
        lora_trainer, "detect_device", lambda preferred="auto": DeviceInfo("cpu", True, False, False)
    )


def _tiny_config(tmp_path, max_steps: int):
    from ravi_varma.config import LoraTrainingConfig

    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i, (w, h) in enumerate([(64, 64), (64, 96), (64, 64), (96, 64)]):
        path = processed / f"{i:04d}.jpg"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)
        rows.append({"image": str(path), "text": f"<rvvarma>, test image {i}", "width": w, "height": h})
    manifest = processed / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    return LoraTrainingConfig.load(
        {
            "base_model": TINY_MODEL,
            "dataset": {"manifest_file": str(manifest), "resolution": 64},
            "lora": {"rank": 4, "alpha": 4},
            "optimizer": {"lr_warmup_steps": 1, "use_8bit_adam": False},
            "training": {
                "max_train_steps": max_steps,
                "gradient_accumulation_steps": 2,
                "mixed_precision": "no",
                "gradient_checkpointing": False,
                "enable_xformers": False,
            },
            "checkpointing": {"output_dir": str(tmp_path / "lora"), "checkpointing_steps": 2, "checkpoints_total_limit": 2},
            "logging": {"logging_dir": str(tmp_path / "logs"), "report_to": "none", "log_every_n_steps": 1},
            "validation": {
                "validation_prompts": ["<rvvarma>, test"],
                "validation_steps": 2,
                "run_at_start": True,
                "num_inference_steps": 2,
                "width": 64,
                "height": 64,
            },
        }
    )


def test_train_checkpoint_export_and_resume(tmp_path, cpu_only):
    from ravi_varma.training.lora_trainer import LoraTrainer

    out = LoraTrainer(_tiny_config(tmp_path, max_steps=4)).train()

    # Final LoRA + self-describing run files.
    assert (out / "pytorch_lora_weights.safetensors").exists()
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["optimizer_steps"] == 4
    assert summary["num_training_samples"] == 4
    assert (out / "training_config.yaml").exists()

    # Loss history: one row per optimizer step, all finite.
    lines = (out / "loss_history.csv").read_text().strip().splitlines()
    assert lines[0] == "step,loss,lr" and len(lines) == 5
    assert all(np.isfinite(float(line.split(",")[1])) for line in lines[1:])

    # Every checkpoint carries a loadable LoRA file; old ones are pruned.
    checkpoints = sorted(p.name for p in out.glob("checkpoint-*"))
    assert checkpoints == ["checkpoint-2", "checkpoint-4"]
    assert all((out / c / "pytorch_lora_weights.safetensors").exists() for c in checkpoints)

    # Validation images at step 0 (baseline) and at every validation step.
    val = tmp_path / "logs" / "validation_images"
    assert sorted(p.name for p in val.iterdir()) == ["step_00000", "step_00002", "step_00004"]
    assert (val / "step_00004" / "grid.jpg").exists()

    # Resuming with a higher step budget continues from checkpoint-4.
    out = LoraTrainer(_tiny_config(tmp_path, max_steps=6)).train()
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["optimizer_steps"] == 6
    assert len((out / "loss_history.csv").read_text().strip().splitlines()) == 7


def test_exported_lora_loads_into_pipeline(tmp_path, cpu_only):
    from diffusers import StableDiffusionPipeline

    from ravi_varma.generation.pipeline import resolve_lora_weights
    from ravi_varma.training.lora_trainer import LoraTrainer

    out = LoraTrainer(_tiny_config(tmp_path, max_steps=2)).train()
    weights = resolve_lora_weights(out)
    pipe = StableDiffusionPipeline.from_pretrained(TINY_MODEL, safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(str(weights.parent), weight_name=weights.name)
    image = pipe("<rvvarma>, test", num_inference_steps=2, width=64, height=64, output_type="np",
                 generator=torch.Generator().manual_seed(0)).images[0]
    assert image.shape == (64, 64, 3) and np.isfinite(image).all()


def test_only_lora_parameters_are_trainable(tmp_path, cpu_only):
    from ravi_varma.training.lora_trainer import LoraTrainer

    trainer = LoraTrainer(_tiny_config(tmp_path, max_steps=1))
    trainer._weight_dtype = torch.float32
    trainer._load_models()
    params = trainer._apply_lora("cpu")
    trainable_names = [n for n, p in trainer._unet.named_parameters() if p.requires_grad]
    assert trainable_names and all("lora" in n for n in trainable_names)
    assert sum(p.numel() for p in params) < 0.2 * sum(p.numel() for p in trainer._unet.parameters())
    assert not any(p.requires_grad for p in trainer._vae.parameters())
    assert not any(p.requires_grad for p in trainer._text_encoder.parameters())
