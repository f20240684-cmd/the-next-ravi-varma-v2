<!-- Extracted from the technical handbook (docs/handbook/src/); edit the source files and rerun build_handbook.py. Commit e2cae62. -->

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
