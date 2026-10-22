# 3. Architecture

## 3.1 What kind of system this is

The Next Ravi Varma is a **single-process Python application plus a set of batch scripts**.

- **Not here:** a server fleet, a database, authentication, a job queue or a deployment.
- **The "database"** is a set of versioned files (JSONL metadata, manifests, caption sidecars, generation sidecars).
- **The "backend"** is one Python class, `RaviVarmaGenerator`. It is used both by the CLI and by a local Gradio web app.

Keeping this in mind avoids describing infrastructure that isn't there.

| Layer | Present? | Where |
|---|---|---|
| Front end | Yes: Gradio (server-rendered components + CSS) | `app/app.py`, `app/components.py`, `app/styles.css` |
| Back end / business logic | Yes: a Python library | `src/ravi_varma/` |
| Batch pipelines | Yes: CLI scripts | `scripts/` |
| Database | **No.** Versioned files instead | `data/metadata/`, `data/processed/manifest.jsonl`, `outputs/` |
| Authentication / authorization | **No.** Local single-user tool bound to `127.0.0.1` | `demo.launch(server_port=…)` |
| Background jobs | **No.** Requests run synchronously; Gradio queues events | — |
| External services | Hugging Face Hub (models), Wikimedia Commons and Wellcome Collection (data), optional LLM API | see §3.7 |
| Deployment | Local machine + Colab for training + GitHub Actions for CI | §3.8 |

## 3.2 High-level system architecture

```mermaid
flowchart TB
  SRC["Open data sources<br/>Wikimedia Commons API · Wellcome Collection IIIF"]
  DATA["Dataset pipeline (local CPU / MPS)<br/>download_dataset.py → validate_dataset.py → generate_captions.py → prepare_dataset.py"]
  TRAIN["Training on a Colab T4 (CUDA)<br/>train_lora.py · LoraTrainer"]
  W[("LoRA weights<br/>pytorch_lora_weights.safetensors · 12.8 MB")]
  GEN["RaviVarmaGenerator<br/>(Mac MPS or CUDA)"]
  PE["PromptExpansionEngine"]
  LLM["Optional LLM API<br/>(OpenAI-compatible)"]
  CLI["scripts/generate.py"]
  APP["app/app.py (Gradio)"]
  OUT["outputs/generated<br/>png + json sidecar"]
  EV["evaluate.py · eval_controlnet.py"]
  HF[("Hugging Face Hub<br/>SD1.5, VAE, BLIP, ControlNets, CLIP")]
  SRC --> DATA
  DATA -->|"manifest.jsonl + images"| TRAIN
  TRAIN --> W
  W --> GEN
  CLI --> GEN
  APP --> GEN
  CLI --> PE
  APP --> PE
  PE -.->|optional| LLM
  GEN --> OUT
  OUT --> EV
  HF -.->|models| DATA
  HF -.->|models| TRAIN
  HF -.->|models| GEN
```

**How to read it:**
- Data flows left to right: **sources → dataset → training → weights → generation → evaluation**.
- Dashed arrows are network downloads or optional calls.
- **Two interfaces share one core.** `scripts/generate.py` and `app/app.py` both construct a `RaviVarmaGenerator` and a `PromptExpansionEngine`, so CLI and app behave identically.
- **The weights are the only hand-off between training and inference.** Training writes `checkpoints/lora/ravi_varma_v3/`; generation reads it through `configs/generation.yaml` → `model.lora_path`. Training on a T4 and generating on a Mac therefore needs only one 12.8 MB file to cross machines.
- **Evaluation reads files, not objects.** It consumes the images and `.json` sidecars written by generation, so any image the project produced can be re-scored later.

## 3.3 Module and component relationships

```mermaid
flowchart LR
  config["config.py<br/>pydantic models"]
  subgraph data["ravi_varma.data"]
    direction TB
    validation["validation.py"]
    captions["captions.py"]
    dataset["dataset.py"]
    metadata["metadata.py"]
  end
  subgraph training["ravi_varma.training"]
    direction TB
    trainer["lora_trainer.py"]
    callbacks["callbacks.py"]
  end
  subgraph generation["ravi_varma.generation"]
    direction TB
    pipeline["pipeline.py<br/>RaviVarmaGenerator"]
    prompt["prompt_engine.py"]
    controlnet["controlnet.py"]
  end
  subgraph evaluation["ravi_varma.evaluation"]
    direction TB
    runeval["run_eval.py"]
    clip["clip_score.py"]
    stylesim["style_similarity.py"]
    report["report.py"]
    pose["pose_score.py"]
  end
  subgraph utils["ravi_varma.utils"]
    direction TB
    image["image.py"]
    device["device.py"]
    logging["logging.py"]
  end
  validation --> image
  dataset --> image
  captions --> image
  validation --> config
  dataset --> config
  trainer --> dataset
  trainer --> callbacks
  trainer --> device
  trainer --> config
  pipeline --> controlnet
  pipeline --> device
  pipeline --> config
  prompt --> captions
  runeval --> clip
  stylesim --> clip
  report --> clip
  report --> stylesim
  pose --> controlnet
  data --> logging
  generation --> logging
```

**Dependency rules (Implemented):**
- **`utils` depends on nothing project-specific.** `config.py` depends only on pydantic and YAML.
- **`data` never imports `training` or `generation`.** It can run without PyTorch: heavy imports are lazy (`captions.py`) or absent (`validation.py`). That's why validation and preparation work on a machine without the ML stack.
- **`generation` doesn't import `training`.** The only coupling is the weights file format.
- **`prompt_engine` imports `captions.clip_token_counter`.** Captions and prompts are both measured with the same tokenizer as the SD1.5 text encoder, so the 77-token rule is enforced identically at training time and generation time.
- **`pose_score` reuses the cached OpenPose detector** from `controlnet.py` (`_openpose_detector`), so the detector loads once per process.

## 3.4 The dataset pipeline

```mermaid
flowchart TB
  A["data/raw/NNNN.jpg<br/>+ metadata.jsonl"] --> B{"DatasetValidator.validate"}
  B -->|"corrupt / < 384 px /<br/>upscaled original"| X1["excluded + reason"]
  B -->|curation.exclude| X2["excluded: curated"]
  B -->|"sha256 / curated group /<br/>dHash Hamming ≤ 40"| X3["duplicates:<br/>keep largest"]
  B --> C["accepted list"]
  C --> D["VLMCaptioner<br/>BLIP sentence + 12 VQA Qs"]
  D --> E["data/captions/NNNN.json<br/>+ flat caption in metadata"]
  E --> F["prepare_dataset:<br/>trim borders → nearest bucket"]
  F -->|holdout| G["data/reference/ + reference.jsonl (12)"]
  F --> H["data/processed/ + manifest.jsonl (182)"]
```

The same `DatasetValidator` runs inside `validate_dataset.py`, `generate_captions.py` and `prepare_dataset.py`
(**Confirmed**, module docstring). All three stages therefore agree on exactly which files are in the dataset; a
curation edit in `configs/dataset.yaml` propagates everywhere.

## 3.5 Training pipeline

```mermaid
flowchart TD
  M["manifest.jsonl"] --> DS["RaviVarmaImageCaptionDataset<br/>(keep_aspect)"]
  DS --> BS["BucketBatchSampler<br/>(same-size batches)"]
  BS --> CC["CaptionCollator<br/>tokenize, 10% → trigger only"]
  CC --> STEP
  subgraph STEP["Training step (Accelerate, fp16)"]
    V["VAE.encode → latents × 0.18215"] --> N["add noise (+ offset 0.05)<br/>at random timestep t"]
    T["CLIP text encoder (frozen)"] --> U
    N --> U["UNet + LoRA (r=16)<br/>predict noise"]
    U --> L["MSE × Min-SNR weight (γ=5)"]
    L --> B["backward → only LoRA A,B<br/>AdamW, grad-accum 4"]
  end
  STEP --> CK["every 250 steps:<br/>checkpoint + LoRA file<br/>+ 3 validation images"]
  STEP --> FIN["final: pytorch_lora_weights.safetensors<br/>training_config.yaml, summary, loss CSV"]
```

**Frozen vs trained:** the VAE, text encoder and UNet base weights are frozen and cast to fp16. Only the LoRA matrices
are trained, in fp32. `_apply_lora` asserts this: it raises if any trainable parameter name lacks "lora".

## 3.6 Inference: one loaded model, many pipelines

```mermaid
flowchart TD
  CFG["configs/generation.yaml"] --> LOAD["RaviVarmaGenerator.load()"]
  LOAD --> P["StableDiffusionPipeline<br/>SD1.5 fp16 + sd-vae-ft-mse<br/>+ v3 LoRA"]
  P -->|"components (shared weights)"| I2I["StableDiffusionImg2ImgPipeline<br/>_redraw: face fix, hires fix"]
  P -->|"components + OpenPose/Canny ControlNet"| CN["StableDiffusionControlNetPipeline<br/>pose control"]
  P -->|"components + Canny ControlNet"| CNI["StableDiffusionControlNetImg2ImgPipeline<br/>_photo_redraw: photo mode"]
  REQ["generate(prompt, …)"] --> SEL{"init_image? control_image?"}
  SEL -->|photo| CNI
  SEL -->|pose| CN
  SEL -->|text| P
  CNI --> POST
  CN --> POST
  P --> POST["black-image guard →<br/>hires (optional) → face fix (optional)"]
  POST --> SAVE["_save: png + json sidecar<br/>(+ _control.png)"]
```

**The key design point:**
- SD1.5 is loaded **once**. Every other pipeline is assembled from `self._pipeline.components`, the same UNet (with the LoRA already injected), VAE and text encoder; only the small ControlNet is added.
- Without this, the original scaffold loaded a second full SD1.5 for ControlNet, which an 8 GB Mac cannot hold (**Confirmed**, the comment in `_load_controlnet_pipeline` and commit `737a8da`).
- ControlNet pipelines are cached per type in `self._controlnet_pipelines`.

## 3.7 External integrations

```mermaid
flowchart TB
  D["download_dataset.py"]
  G["generator · captioner · CLIP · trainer"]
  PE["LLMPromptEngine (optional)"]
  CM["commons.wikimedia.org<br/>categorymembers, imageinfo"]
  WE["api.wellcomecollection.org<br/>images search + IIIF"]
  HUB["huggingface.co<br/>from_pretrained, cached"]
  OA["OpenAI-compatible API<br/>(e.g. Gemini)"]
  AN["Anthropic API"]
  D -->|"GET, retry 429/5xx"| CM
  D -->|"GET, 2048-px renditions"| WE
  G -->|"model download"| HUB
  PE -->|"chat.completions"| OA
  PE -->|"messages.create"| AN
```

- **Wikimedia Commons.** Recursive category walk plus `imageinfo` queries for licence, original size and a 2048-px rendition. 429 responses honour `Retry-After` (`api_get`). Downloads sleep 2 s per file to stay under rate limits.
- **Wellcome Collection.** Image search, then the IIIF `info.json` for the original size. Only `pdm`, `cc-0` and `cc-by` licences are accepted, and the title must mention Varma (`download_wellcome`).
- **Hugging Face Hub.** All models are pulled with `from_pretrained` and cached locally; no token is needed, because all models are public.
- **LLM (optional).** Off unless `.env` sets `PROMPT_EXPANSION_PROVIDER`. Only scene text is sent; failures fall back to the local engine (`PromptExpansionEngine.expand`).

## 3.8 Deployment and environments

```mermaid
flowchart TB
  DEV["Developer Mac (8 GB, MPS)<br/>dataset pipeline, generation, app, tests"] -->|"git push"| GH["GitHub repo"]
  GH -->|"on push / PR"| CI["GitHub Actions (Ubuntu, CPU torch)<br/>pyflakes + pytest"]
  DEV -->|"zip upload (3 MB parts)"| COLAB["Google Colab T4<br/>TAG=v3 bash scripts/colab_train.sh"]
  COLAB -->|"ravi_varma_results.zip"| DEV
  GH -->|"git clone"| EVAL["Evaluator machine<br/>README quick start"]
```

There is **no hosted deployment**. "Deployment" means three environments: the developer laptop, a Colab GPU for
training, and CI. The **proposed future architecture** (not implemented) would add a Hugging Face Space for a public
demo and the Hub for weights (§11).

## 3.9 Request lifecycle in the web app

```mermaid
sequenceDiagram
  participant U as User (browser)
  participant G as Gradio server (app.py)
  participant C as components.run_generation
  participant E as PromptExpansionEngine
  participant R as RaviVarmaGenerator
  U->>G: click "Generate painting" (scene, settings)
  G->>C: call with 14 input values
  C->>C: validate scene not empty, seed 0 → random
  C->>R: set_lora(checkpoint) if changed
  C->>E: expand(scene) if "Expand" checked
  E-->>C: ExpandedPrompt.final_prompt() (≤ 75 tokens)
  C->>R: generate(prompt, size, seed, pose, face_fix …)
  R-->>C: GenerationResult(image, metadata, control_image)
  C-->>G: image, caption HTML, prompt card, details, warning, pose map, gallery
  G-->>U: update 7 output components
```

Each click runs synchronously in the Gradio worker. There is no streaming or progress callback; the user waits about
1–2 minutes on a Mac. Errors raised as `RuntimeError` are caught and shown as a warning instead of crashing the app.
