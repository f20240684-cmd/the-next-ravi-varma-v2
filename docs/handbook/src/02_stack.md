# 2. Technology stack and dependencies

Every entry below was checked against actual imports (`grep` over `src/`, `scripts/`, `app/`, `tests/`). Pinned
versions come from `requirements.txt`. Since commit `dd26570`, packages listed but never imported have been removed:
`datasets`, `scikit-image`, `lpips` and `pytest-mock`.

## 2.1 Summary table

| Technology (version) | Role in this project | Used in | Essential? | Main alternative |
|---|---|---|---|---|
| Python 3.11 | Language | everything | Essential | — |
| PyTorch 2.4.1 | Tensor maths, autograd, devices (CUDA / MPS / CPU) | trainer, generator, CLIP | Essential | JAX (no Diffusers parity) |
| Diffusers 0.30.3 | SD1.5 pipelines, UNet/VAE classes, schedulers, ControlNet pipelines | `generation/pipeline.py`, `training/lora_trainer.py` | Essential | Hand-written sampling loop; ComfyUI (not a library) |
| Transformers 4.44.2 | CLIP tokenizer and text encoder; BLIP caption / VQA models | trainer, `data/captions.py` | Essential | open_clip for CLIP text (but SD expects HF classes) |
| PEFT 0.12.0 | LoRA adapters injected into the UNet (`add_adapter`) | `lora_trainer.py`, `_apply_lora` | Essential | Diffusers' older attention-processor LoRA; hand-written LoRA |
| Accelerate 0.34.2 | Mixed precision, gradient accumulation, checkpoint `save_state` / `load_state` | `lora_trainer.py`, `train` | Essential for training | A plain PyTorch loop with `torch.cuda.amp` |
| safetensors 0.4.5 | Safe, fast weight files (`pytorch_lora_weights.safetensors`) | via Diffusers / PEFT | Essential | pickle `.bin` (unsafe to load) |
| torchvision 0.19.1 | Training-time transforms (resize, flip, normalise) | `data/dataset.py`, `__getitem__` | Convenient | PIL + NumPy by hand |
| Pillow 10.4.0 | Image IO, resizing, the decompression-bomb limit | everywhere | Essential | OpenCV |
| OpenCV (headless) 4.10 | Canny edges, Haar face and eye detection | `generation/controlnet.py`, `pipeline.py`, `detect_faces` | Essential for those features | scikit-image (Canny); MediaPipe / RetinaFace (faces) |
| NumPy 1.26.4 | Arrays: hashes, borders, embeddings, PCK | utils, evaluation | Essential | — |
| open-clip-torch 2.26.1 | CLIP ViT-B/32 for **evaluation** metrics | `evaluation/clip_score.py` | Essential for evaluation | HF `CLIPModel` |
| controlnet-aux 0.0.9 | OpenPose skeleton detector | `generation/controlnet.py`, `evaluation/pose_score.py` | Essential for pose features | MediaPipe Pose, DWPose |
| Gradio 4.44.0 (+ fastapi 0.115.0 pin) | Web UI | `app/` | Essential for the app | Streamlit (named in the proposal), a FastAPI + JS front end |
| pydantic 2.9.1 | Typed, validated config models | `src/ravi_varma/config.py` | Convenient but valuable | dataclasses + manual checks |
| PyYAML 6.0.2 | Config and lexicon files | config, prompt engine | Essential | TOML / JSON |
| requests 2.32.3 | Wikimedia and Wellcome HTTP APIs | `scripts/download_dataset.py` | Essential for download | httpx, urllib |
| python-dotenv 1.0.1 | Loads `.env` for the optional LLM keys | `scripts/generate.py` | Convenient | `export` in the shell |
| openai 1.55.3 / anthropic 0.34.2 | Optional LLM prompt expansion | `prompt_engine.py`, `LLMPromptEngine` | Optional | raw HTTP |
| tensorboard 2.17.1 | Optional training dashboards (loss is also always written to CSV) | `training/callbacks.py`, `LossLogger` | Optional | the CSV alone |
| tqdm | Progress bars (transitively used by libraries) | — | Convenient | — |
| pytest 8.3.3, pyflakes 3.2.0 | Tests and lint | `tests/`, `Makefile`, CI | Essential for quality | unittest; ruff |
| GitHub Actions | CI on every push | `.github/workflows/tests.yml` | Valuable | none / GitLab CI |
| Google Colab (T4, runtime 2026.07) | Training hardware | `scripts/colab_train.sh` | Practically essential | any CUDA box, Kaggle |

**External models** (downloaded from the Hugging Face Hub at first use):
- `stable-diffusion-v1-5/stable-diffusion-v1-5`: the base model.
- `stabilityai/sd-vae-ft-mse`: the decoder.
- `Salesforce/blip-image-captioning-large` and `Salesforce/blip-vqa-base`: captioning.
- `lllyasviel/control_v11p_sd15_openpose` and `…_canny`: ControlNets.
- `lllyasviel/ControlNet` (Annotators): OpenPose weights.
- OpenAI CLIP ViT-B/32 via open_clip: evaluation.
- `hf-internal-testing/tiny-stable-diffusion-pipe`: the 1 MB test model.

## 2.2 Notes on the important choices

**Diffusers + PEFT + Accelerate.** The trainer mirrors the structure of Diffusers' official `train_text_to_image_lora`
example, which is **Confirmed** in the docstring of `lora_trainer.py`. Using the reference stack means:
- `pipe.load_lora_weights()` can load the exported file directly
- schedulers can be swapped by name (`SCHEDULER_MAP` in `pipeline.py`)
- ControlNet and img2img pipelines can be built from the same components (`StableDiffusionControlNetPipeline(**components, controlnet=…)`)

*Trade-off:* the pinned versions (Diffusers 0.30.3, PEFT 0.12.0) are older than current releases. They are stable
together, but upgrading needs care (§10).

**open_clip vs Transformers CLIP for evaluation.** Generation uses the CLIP text encoder *inside* SD1.5, which is
ViT-L/14 via Transformers. Evaluation uses a *different* CLIP, **ViT-B/32 via open_clip**.

*Inferred rationale:* an independent, smaller judge avoids scoring the model with its own conditioning encoder, and
is cheap on CPU. *Consequence:* scores aren't comparable with papers that use ViT-L/14 CLIP-score.

**Gradio instead of Streamlit.** The proposal names "Gradio or Streamlit". Gradio's event model maps one button to one
Python function, so `run_generation(…)` is just the generator call plus HTML formatting.

*Pitfall found:* Gradio 4.44 crashes with starlette 1.x (`TypeError: unhashable type: 'dict'`), hence the explicit
`fastapi==0.115.0` pin (**Confirmed**, comment in `requirements.txt`).

**pydantic config models.** `config.py` declares `DatasetConfig`, `LoraTrainingConfig` and `GenerationConfig`, with
defaults and validators; for example `GenerationParamsConfig._multiple_of_8` rejects widths that aren't multiples of
8. A typo in a YAML key falls back to the default rather than crashing, which is a mixed blessing (§10).

**OpenCV Haar cascades for faces.** They are already installed with OpenCV and run on CPU in milliseconds. They're
weaker than a neural detector on profiles and crowds. A second Haar cascade for **eyes** filters false positives such
as medals and folds of cloth (`detect_faces`, **Confirmed** by its docstring and commit `3958585`).

## 2.3 What would change if a component were replaced

| Replace | Impact |
|---|---|
| SD1.5 → SDXL | Retrain the LoRA (weights don't transfer). New pipeline classes, two text encoders, ~3× memory (no longer fits the 8 GB Mac), SDXL ControlNets |
| PEFT LoRA → full fine-tune | ~4 GB checkpoints, high overfitting risk with 182 images, needs far more GPU memory |
| Gradio → Streamlit | Rewrite `app/` (about 500 lines); the generator is untouched, because UI and logic are already separate |
| open_clip → HF CLIP | Change `evaluation/clip_score.py` only; all metric values would shift, so earlier results must be recomputed |
| Haar → DNN face detector | Change `detect_faces` only; add a dependency; fewer misses on profiles |
| Colab → local GPU | No code change; `scripts/colab_train.sh` runs on any CUDA Linux box |
