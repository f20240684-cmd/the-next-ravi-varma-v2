# 1. Project overview

## 1.1 The problem

Raja Ravi Varma (1848–1906) is the painter who defined how Indian epics *look* in popular imagination: academic
European oil technique (soft, modelled skin; rich drapery; theatrical lighting and staging) applied to Mahabharata,
Ramayana and Puranic subjects. His paintings are in the public domain.

A general text-to-image model does not reproduce this. The repository documents what plain **Stable Diffusion 1.5**
produces for prompts such as "royal Indian woman in a palace, classical oil painting" (`docs/RESULTS.md`, *Base-model
baseline*; images in `outputs/generated/baseline/`):
- a glossy, frontal, saturated "calendar art" look with hard lighting
- **painted gilt frames in 5 of 15 images**, even though "frame" is in the negative prompt
- ambiguous identities in multi-figure mythological scenes

The project asks three questions (**Confirmed**: README and the proposal `The_Next_Ravi_Varma_AI_Project_Proposal.docx`):
1. Can a **small, curated corpus** (under 200 paintings) teach a diffusion model one artist's visual language?
2. Can this be done on **free or modest hardware**: a Google Colab T4 for training, an 8 GB Apple-silicon laptop for generation?
3. Can the result be **measured honestly** against the base model, rather than judged only by eye?

## 1.2 Who it is for

| User | What they need | How the project serves it |
|---|---|---|
| The student team (primary) | A working, explainable system for a 4-milestone course project | Scripts per stage, docs, a viva guide, this handbook |
| An evaluator / reviewer | To clone, run and verify claims independently | Shipped v3 LoRA, CI, a README quick start, results traceable to files |
| An end user / demo audience | To make Varma-style images from text, a pose or a photo | The Gradio web app (`app/app.py`) and CLI (`scripts/generate.py`) |
| A future developer | To retrain or extend the model | Config-driven pipeline, `scripts/colab_train.sh`, tests |

## 1.3 Core use cases

1. **Text → painting:** type a scene; get a Ravi Varma–style image.
2. **Story → better prompt** (Milestone 3): write "Krishna guiding Arjuna in his chariot"; the system adds each character's standard iconography and fits the prompt to CLIP's 77-token limit.
3. **Pose control** (Milestone 4): give a reference image; the painting copies its pose (ControlNet OpenPose) or its outlines (ControlNet Canny).
4. **Photo → painting:** repaint a photo in Varma's style while keeping its composition.
5. **Rebuild the dataset and retrain the model** from committed metadata.
6. **Evaluate** a model: paired CLIP metrics against the base model and held-out paintings; pose adherence (PCK).

## 1.4 Scope and non-goals

**In scope (Implemented):** the dataset pipeline, LoRA training, the generation features above, evaluation, the web app.

**Explicit non-goals and things that are not implemented:**
- **Authentic reproduction.** Every output is labelled AI-generated and "Ravi Varma–inspired" (generation metadata `disclaimer` field, app footer, README).
- **SDXL training.** `configs/lora_sdxl.yaml` exists, but `LoraTrainer._load_models` raises `NotImplementedError` for `model_family: sdxl`. That's scaffolding, not a feature.
- **Production serving.** There are no accounts, no database, no deployment and no multi-user concurrency handling. The Gradio app is a local single-user tool.
- **A human evaluation study.** `qualitative_review.csv` provides an empty rubric; no human ratings exist yet.

## 1.5 Implementation status (verified 22 Oct 2026)

| Capability | Status | Evidence |
|---|---|---|
| Dataset download with provenance (Commons + Wellcome) | Implemented, used | `scripts/download_dataset.py`; `data/metadata/metadata.jsonl` (300 records) |
| Validation, curation, de-duplication | Implemented, tested | `src/ravi_varma/data/validation.py`; `tests/test_curation.py`, `tests/test_dataset.py` |
| BLIP + VQA captioning | Implemented, used | `src/ravi_varma/data/captions.py`; 194 sidecars in `data/captions/` |
| Aspect-bucket preparation, hold-out split | Implemented, tested | `src/ravi_varma/data/dataset.py`; `data/processed/manifest.jsonl` (182), `data/reference/` (12) |
| LoRA training (SD1.5) with checkpoint/resume | Implemented, tested on a tiny model, **run on Colab T4** | `src/ravi_varma/training/`; `checkpoints/lora/ravi_varma_v3/training_summary.json` |
| Text→image CLI and app | Implemented, run on Mac and T4 | `scripts/generate.py`, `src/ravi_varma/generation/pipeline.py` |
| Prompt expansion (lexicon + optional LLM) | Implemented, tested; **no measurable CLIP gain** | `generation/prompt_engine.py`; `outputs/evaluations/m3_raw_local_llm.json` |
| ControlNet pose / edge control | Implemented, evaluated | `generation/controlnet.py`, `scripts/eval_controlnet.py`; `outputs/evaluations/m4_controlnet/summary.json` |
| Face fix, hires fix | Implemented; hires needs a GPU with > 8 GB | `RaviVarmaGenerator._fix_faces`, `generate(hires_scale=…)` |
| Photo → painting | Implemented, run on T4 and Mac | `generate(init_image=…)`, `_photo_redraw`; `outputs/evaluations/photo_to_varma.jpg` |
| Paired evaluation, pose PCK | Implemented, tested | `evaluation/run_eval.py`, `evaluation/pose_score.py` |
| Gradio web app (5 tabs) | Implemented, browser-tested | `app/app.py`, `app/components.py`; `docs/screenshots/` |
| CI | Implemented, passing | `.github/workflows/tests.yml` (119 passed, 2 skipped on Ubuntu) |
| SDXL | **Not implemented** | `lora_trainer.py`, `_load_models` raises `NotImplementedError` |

## 1.6 Constraints and assumptions

- **Hardware.** Training needs CUDA (fp16 on a 16 GB T4). Generation targets an **8 GB Apple-silicon Mac** (MPS, fp16), a constraint that shaped several designs (§5, D-13, D-14).
- **Free tiers only.** No paid API is required. The optional LLM mode works with any OpenAI-compatible endpoint (Gemini's free tier was used).
- **Data licence.** Only public-domain or openly licensed reproductions, with provenance recorded per image.
- **Small data.** 182 training images (v3). Every design choice about capacity (LoRA rank, steps) has to respect that.
- **Assumption (Inferred):** the evaluator has internet access on first run, because the base model (about 4 GB), CLIP and the ControlNets download from the Hugging Face Hub.

## 1.7 What makes it technically interesting

1. **Data quality over quantity, made auditable.** Resolution is checked against the *original* scan; 26 upscaled copies were caught that way. Duplicates are found by exact hash, perceptual hash and CLIP-assisted human review. Every exclusion has a recorded reason.
2. **Teaching a style through captions.** The captions describe content in detail, so the trigger token `<rvvarma>` ends up holding only the *style*. Unwanted artefacts (frames, printed text) are named in captions so the trigger word doesn't absorb them.
3. **One loaded model, many pipelines.** txt2img, img2img (face and hires fixes), ControlNet, and ControlNet-img2img (photo mode) all share one set of weights. That's what makes everything fit in 8 GB.
4. **Honest evaluation.** The evaluation is paired against the base model, uses held-out references, checks for memorisation, and adds a pose-adherence metric (PCK). Null results are reported as null results.

## 1.8 A realistic session

A student preparing a demo runs:

```bash
source .venv/bin/activate
python scripts/generate.py --prompt "Saraswati playing the veena by a lotus lake" --expand --dry-run
```

The output is the expanded prompt, printed without loading any model:
`<rvvarma>, Saraswati playing the veena by a lotus lake, Saraswati in a white sari playing the veena, a swan beside her, a lotus lake, …`

Then they render it, with faces sharpened:

```bash
python scripts/generate.py --prompt "Saraswati playing the veena by a lotus lake" --expand --seed 42 --width 448 --height 640 --face-fix
```

The image lands in `outputs/generated/` with a `.json` sidecar recording the prompt, seed, steps, LoRA file, decoder
and timing, so the exact image can be regenerated. For the audience they open `python app/app.py` and use the
**Generate** and **Photo → Varma** tabs.
