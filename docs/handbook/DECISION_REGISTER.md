<!-- Extracted from the technical handbook (docs/handbook/src/); edit the source files and rerun build_handbook.py. Commit e2cae62. -->

# 5. Decision register

**Sources.** Each decision cites its evidence. `docs/DECISIONS.md` records the team's stated reasons, but it was
written *after* most of the choices were made (it was added late in the project), so it is a record of the
rationale, not a contemporaneous log. Where neither that document nor a commit message gives a reason, the rationale
is marked **Inferred**.

**Confidence scale:**
- **High:** documented, and the code or measurements agree.
- **Medium:** documented, but not measured.
- **Low:** inferred.

## Register summary

| ID | Decision | Confidence |
|---|---|---|
| D-01 | SD1.5 as the base model, not SDXL | High |
| D-02 | LoRA instead of full fine-tuning / DreamBooth / textual inversion | High |
| D-03 | Rank 16, alpha 16, attention projections only | Medium |
| D-04 | A trigger token `<rvvarma>` in every caption | High |
| D-05 | Only Wikimedia Commons + Wellcome Collection as sources | High |
| D-06 | Validate resolution against the original scan | High |
| D-07 | Aggressive de-duplication (keep the sharpest copy) | High |
| D-08 | Keep the chromolithograph prints, tagged | Medium |
| D-09 | Hold out 12 paintings; no validation split | Medium |
| D-10 | Aspect-ratio buckets | High |
| D-11 | Name artefacts in captions; template BLIP + VQA captions in 77 tokens | High |
| D-12 | Train on Colab; pinned runtime; smoke test first | High |
| D-13 | Share one loaded model across all pipelines | High |
| D-14 | Mac settings: fp16, no attention slicing, CPU seeds, `empty_cache` | High |
| D-15 | Fine-tuned VAE as decoder | Medium |
| D-16 | v3 training additions: Min-SNR, noise offset, caption dropout | Medium |
| D-17 | Local lexicon first, LLM optional, priority assembly | High |
| D-18 | OpenPose as the main control; PCK as its metric | High |
| D-19 | Face fix as a second pass; hires fix only on GPU | Medium |
| D-20 | Photo mode = img2img + Canny ControlNet lock | High |
| D-21 | Gradio, styled from a Figma design | Medium |
| D-22 | Paired, held-out CLIP evaluation; report null results | High |
| D-23 | pydantic config models with defaults | Low |
| D-24 | Ship the final 12.8 MB LoRA in git | High |

---

### D-01 SD1.5 as the base model

| Field | Content |
|---|---|
| Decision | Fine-tune `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Evidence | `configs/lora_sd15.yaml`; SDXL raises `NotImplementedError` in `LoraTrainer._load_models`; DECISIONS §1.1 |
| Problem | Hardware: a free T4 for training, an 8 GB Mac for generation |
| Rationale | SD1.5 trains in ~1 h on a T4 and generates in ≈3.5 GB; SDXL needs ~3× the memory. Mature SD1.5 ControlNets exist |
| Alternatives | SDXL; DiT-based models (SD3, Flux); training from scratch |
| Trade-offs | ~512 px native resolution → soft faces in full-body shots |
| Consequences | The face fix and hires fix were needed (D-19) |
| Reconsider when | A ≥ 24 GB GPU is available for training *and* inference |
| Confidence | High |

### D-02 LoRA instead of full fine-tuning

| Field | Content |
|---|---|
| Decision | Train PEFT LoRA adapters only; freeze the 860 M UNet |
| Evidence | `_apply_lora` asserts that only LoRA parameters are trainable; a 12.8 MB weights file; DECISIONS §1.2 |
| Problem | 182 images would be memorised by a full fine-tune |
| Rationale | 3.2 M parameters (0.37 %); can be switched on and off and scaled |
| Alternatives | Full fine-tune, DreamBooth, textual inversion |
| Trade-offs | Lower capacity, so a subtle style may be under-learned |
| Consequences | **Measured:** max training-image similarity 0.898 < 0.95, so no copying |
| Reconsider when | Thousands of images are available |
| Confidence | High |

### D-03 Rank 16, alpha 16, attention only

| Field | Content |
|---|---|
| Decision | `r=16, alpha=16, dropout=0.05`, targets `to_q, to_k, to_v, to_out.0` |
| Evidence | `training_summary.json`; DECISIONS §1.3 |
| Problem | Choosing the capacity |
| Rationale | A common middle ground; α/r = 1 keeps the update scale neutral; cross-attention ties text to image |
| Alternatives | r 4 / 8 / 32 / 64; adding feed-forward and conv layers (LoCon) |
| Trade-offs | **Not measured.** No ablation was run |
| Consequences | Results can't say whether another rank would be better |
| Reconsider when | Time for a 3-run ablation (~3 GPU-hours) |
| Confidence | Medium |

### D-04 Trigger token

| Field | Content |
|---|---|
| Decision | Every caption starts with `<rvvarma>, `; generation adds it if missing |
| Evidence | `CaptionCollator`, `test_trigger_token_added_once`, DECISIONS §1.4 |
| Problem | Separating style from content |
| Rationale | Whatever the captions don't describe gets attached to the only word present in every caption |
| Alternatives | No trigger (the style is always on); a rare real word (e.g. "sks") |
| Trade-offs | CLIP splits it into 5 tokens, costing budget |
| Consequences | A clean on/off comparison; caption dropout (D-16) strengthens it |
| Reconsider when | Never, for this design |
| Confidence | High |

### D-05 Data sources

| Field | Content |
|---|---|
| Decision | Wikimedia Commons + Wellcome Collection only |
| Evidence | `download_dataset.py`; licence field in `metadata.jsonl`; DECISIONS §2.1 |
| Problem | Legally usable, high-resolution images with provenance |
| Rationale | Open APIs, per-image licences, public domain (Varma died in 1906). Google Arts & Culture's terms forbid downloading; other museum APIs had no relevant works |
| Alternatives | Scraping galleries; Google Arts & Culture; museum APIs |
| Trade-offs | Variable scan quality; prints mixed with oils |
| Consequences | D-06, D-07 and D-08 were needed |
| Reconsider when | A museum releases high-resolution open scans |
| Confidence | High |

### D-06 Resolution checked against the original

| Field | Content |
|---|---|
| Decision | Validate `source_width/height`, not the local file size |
| Evidence | `DatasetValidator`; DECISIONS §2.2; commit history (v2) |
| Problem | 26 files were upscaled to exactly 384 px from originals as small as 200 px |
| Rationale | Upscaling adds blur, not detail |
| Alternatives | A blur / sharpness metric |
| Trade-offs | Needs metadata at download time |
| Consequences | The second download used `--min-side 800`; median short side 2,400 px for the new images |
| Reconsider when | — |
| Confidence | High |

### D-07 De-duplication

| Field | Content |
|---|---|
| Decision | SHA-256 + curated groups + dHash ≤ 40; keep the largest copy |
| Evidence | `configs/dataset.yaml` (`duplicate_groups`); `tests/test_curation.py`; DECISIONS §2.4 |
| Problem | Up to 7 copies of one painting (the Saraswati oleograph) |
| Rationale | Duplicates are learned several times as strongly, which reduces variety |
| Alternatives | Exact hash only; CLIP clustering alone |
| Trade-offs | O(n²) comparison; human review time |
| Consequences | Groups are auditable in config |
| Reconsider when | The dataset passes ~10k images (use LSH) |
| Confidence | High |

### D-08 Keep the prints, tagged

| Field | Content |
|---|---|
| Decision | 29 accepted chromolithographs kept, tagged `medium: chromolithograph` |
| Evidence | `metadata.jsonl`; DECISIONS §2.6 |
| Problem | Prints may flatten colour, but they are genuine compositions |
| Rationale | The dataset is small; tagging allows dropping them later in one line |
| Alternatives | Exclude all prints |
| Trade-offs | Their effect on style is **unknown** |
| Consequences | An open question (Appendix A) |
| Reconsider when | An ablation run without prints |
| Confidence | Medium |

### D-09 Hold-out of 12

| Field | Content |
|---|---|
| Decision | 12 accepted paintings never trained on; no validation split |
| Evidence | `configs/dataset.yaml` → `holdout`; `data/reference/` |
| Problem | Style must be measured without rewarding memorisation |
| Rationale | Every training image counts at n < 200; 12 references suffice for mean similarity |
| Alternatives | 80/20 split; k-fold |
| Trade-offs | High variance; no validation loss |
| Consequences | Checkpoints are judged by eye from fixed prompts |
| Reconsider when | The dataset grows past ~500 |
| Confidence | Medium |

### D-10 Aspect buckets

| Field | Content |
|---|---|
| Decision | Resize to the nearest ~512²-area bucket (multiples of 64); batch per bucket |
| Evidence | `BucketBatchSampler`; 122 / 182 images in 448×640 |
| Problem | Square crops cut off heads and feet |
| Rationale | Most canvases are portrait format |
| Alternatives | Centre crop; padding |
| Trade-offs | Smaller effective batches per bucket |
| Consequences | 448×640 became the default generation size |
| Reconsider when | — |
| Confidence | High |

### D-11 Caption strategy

| Field | Content |
|---|---|
| Decision | BLIP sentence + 12 gated VQA answers + palette + title + artefact notes, ≤ 69 + 6 tokens |
| Evidence | `captions.py`; `tests/test_captions.py`; DECISIONS §2.9–2.11 |
| Problem | v1 learned gilt frames; 24 captions overflowed 77 tokens |
| Rationale | Content words absorb content; named artefacts don't leak into the trigger |
| Alternatives | BLIP only; LLaVA; hand-written captions |
| Trade-offs | Generic VQA phrasing; BLIP can't name deities |
| Consequences | **Measured:** frames 5/15 → 0/15 |
| Reconsider when | A strong local VLM fits the hardware |
| Confidence | High |

### D-12 Training on Colab

| Field | Content |
|---|---|
| Decision | Train only on a Colab T4, with a pinned runtime, no bitsandbytes / xformers, and a smoke test first |
| Evidence | `scripts/colab_train.sh`, `requirements-colab.txt`; DECISIONS §3.1–3.3 |
| Problem | No CUDA on the Mac; Colab package churn |
| Rationale | A free GPU; the smoke test caught a crash in 1 minute instead of 1 hour |
| Alternatives | Kaggle; a paid cloud GPU; training on MPS |
| Trade-offs | Session limits; manual file transfer |
| Consequences | `TAG` folders separate runs |
| Reconsider when | A local CUDA GPU is available |
| Confidence | High |

### D-13 One loaded model, many pipelines

| Field | Content |
|---|---|
| Decision | Build img2img / ControlNet pipelines from `self._pipeline.components` |
| Evidence | `_load_controlnet_pipeline`, `_redraw`; commit `737a8da`; DECISIONS §6.2 |
| Problem | Two SD1.5 copies don't fit in 8 GB |
| Rationale | Same weights, different sampling loops |
| Alternatives | Separate pipelines; offloading |
| Trade-offs | The pipelines share state (e.g. LoRA scale) |
| Consequences | All features run on the Mac |
| Reconsider when | — |
| Confidence | High |

### D-14 Apple-silicon settings

| Field | Content |
|---|---|
| Decision | fp16 on MPS without attention slicing; CPU `torch.Generator`; `empty_cache` between images |
| Evidence | `resolve_optimizations`; `test_no_auto_attention_slicing_on_mps`; DECISIONS §4.2–4.3 |
| Problem | Slicing + fp16 → NaN (black images); memory filled up |
| Rationale | Observed failures |
| Alternatives | fp32 (2× memory) |
| Trade-offs | Peak memory slightly higher |
| Consequences | A black-image guard raises instead of saving |
| Reconsider when | Torch / Diffusers upgrades |
| Confidence | High |

### D-15 Fine-tuned VAE

| Field | Content |
|---|---|
| Decision | `stabilityai/sd-vae-ft-mse` as decoder |
| Evidence | `configs/generation.yaml`; DECISIONS §4.4 |
| Problem | Mushy faces and detail |
| Rationale | Same latent space, so no retraining |
| Alternatives | The default VAE |
| Trade-offs | +~330 MB download (one-off) |
| Consequences | `vae: null` reproduces older images |
| Reconsider when | — |
| Confidence | Medium (visual, not measured) |

### D-16 v3 training additions

| Field | Content |
|---|---|
| Decision | Min-SNR γ=5, noise offset 0.05, caption dropout 0.1; other hyperparameters unchanged |
| Evidence | `_min_snr_weights`, `CaptionCollator._text`; `test_v3_training_options_run`; DECISIONS §3.5 |
| Problem | Weak v2 style shift |
| Rationale | Published techniques; off by default for backward compatibility |
| Alternatives | Raising LR or steps |
| Trade-offs | **Confounded** with the new data, so effects can't be separated |
| Consequences | **Measured:** style 0.681 → 0.701; diversity 0.119 → 0.092 |
| Reconsider when | Ablations are run |
| Confidence | Medium |

### D-17 Prompt expansion design

| Field | Content |
|---|---|
| Decision | A deterministic lexicon by default; optional grounded LLM; priority assembly ≤ 75 tokens |
| Evidence | `prompt_engine.py`; `tests/test_prompt_engine.py`; DECISIONS §5 |
| Problem | Users don't know the iconography; an earlier version produced ~120-token prompts |
| Rationale | Offline and reliable for a viva; the LLM is optional and falls back |
| Alternatives | LLM only; no expansion |
| Trade-offs | The lexicon covers only 27 characters |
| Consequences | **Measured:** no CLIP gain, reported as such |
| Reconsider when | A larger evaluation shows a gain |
| Confidence | High |

### D-18 OpenPose + PCK

| Field | Content |
|---|---|
| Decision | OpenPose is the main control; Canny optional; PCK@0.1 as the metric |
| Evidence | `eval_controlnet.py`; `m4_controlnet/summary.json`; DECISIONS §6.1, §6.3 |
| Problem | Multi-figure compositions collapse |
| Rationale | OpenPose transfers pose without copying the image; Canny traces it |
| Alternatives | Depth; regional prompting |
| Trade-offs | OpenPose on paintings is imperfect |
| Consequences | **Measured:** PCK 0.17 → 0.86 |
| Reconsider when | — |
| Confidence | High |

### D-19 Face fix and hires fix

| Field | Content |
|---|---|
| Decision | Detect, crop and redraw faces at 512 px; hires ×1.5 only on GPUs > 8 GB |
| Evidence | `_fix_faces`, `detect_faces`; DECISIONS §7.1–7.2 |
| Problem | ~40 px faces |
| Rationale | ADetailer-style second pass, cheaper than a bigger model |
| Alternatives | SDXL; GFPGAN / CodeFormer face restoration |
| Trade-offs | Haar misses profiles; hires doesn't fit the Mac |
| Consequences | The eye check was added after a medal got a face |
| Reconsider when | A DNN detector or more VRAM |
| Confidence | Medium |

### D-20 Photo mode with a Canny lock

| Field | Content |
|---|---|
| Decision | ControlNet-img2img with Canny edges of the photo; neutral prompt; facial-hair terms dropped from the negative |
| Evidence | `_photo_redraw`; DECISIONS §7.3; the Spider-Man failure recorded there |
| Problem | Plain img2img at 0.7 replaced the subject entirely |
| Rationale | Edges hold the structure; the LoRA repaints texture and colour |
| Alternatives | Plain img2img at low strength (barely stylises); IP-Adapter |
| Trade-offs | Costumes reinterpreted |
| Consequences | Composition and likeness kept |
| Reconsider when | — |
| Confidence | High |

### D-21 Gradio, Figma-styled

| Field | Content |
|---|---|
| Decision | Gradio 4.44 with custom CSS from a Figma design; fastapi pinned |
| Evidence | `app/`, `requirements.txt` comment; DECISIONS §6.4 |
| Problem | A demo UI quickly, in Python |
| Rationale | One button = one function; no JS build |
| Alternatives | Streamlit; FastAPI + React |
| Trade-offs | Less layout control; CSS depends on Gradio internals |
| Consequences | `_theme` filters unsupported keys via `inspect` |
| Reconsider when | Public deployment |
| Confidence | Medium |

### D-22 Evaluation philosophy

| Field | Content |
|---|---|
| Decision | Paired vs base, held-out references, memorisation and diversity checks; failures reported |
| Evidence | `run_eval.py` docstring; DECISIONS §8 |
| Problem | Absolute CLIP scores mix content with style |
| Rationale | Only deltas on identical prompts and seeds are meaningful |
| Alternatives | FID (needs thousands of images); human study only |
| Trade-offs | CLIP is a coarse style judge |
| Consequences | Honest but modest numbers |
| Reconsider when | Human ratings are available |
| Confidence | High |

### D-23 pydantic configs with defaults

| Field | Content |
|---|---|
| Decision | YAML → pydantic models with defaults |
| Evidence | `config.py` |
| Rationale (**Inferred**) | Typed access and validators (e.g. multiple-of-8 sizes) |
| Trade-offs | Unknown keys are ignored silently (a typo falls back to the default) |
| Reconsider when | Always; add `extra="forbid"` (§10) |
| Confidence | Low |

### D-24 Ship the final LoRA in git

| Field | Content |
|---|---|
| Decision | Un-ignore only `checkpoints/lora/ravi_varma_v3/` final files (12.8 MB) |
| Evidence | `.gitignore`; commit `f49f890` |
| Problem | A fresh clone couldn't reproduce results without retraining |
| Rationale | Small enough for git; Git LFS or the Hub would add setup |
| Alternatives | Hugging Face Hub; GitHub release asset; Git LFS |
| Trade-offs | Binary in history |
| Consequences | `checkpoints/README.md` is now stale (Appendix A) |
| Reconsider when | Weights change often, or exceed ~50 MB |
| Confidence | High |
