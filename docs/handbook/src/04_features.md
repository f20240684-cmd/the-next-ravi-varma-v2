# 4. Features

## 4.1 Feature inventory

| # | Feature | Status | Entry point |
|---|---|---|---|
| F1 | Dataset download with provenance | Implemented, used | `scripts/download_dataset.py` |
| F2 | Validation, curation and de-duplication | Implemented, tested | `scripts/validate_dataset.py`, `DatasetValidator` |
| F3 | BLIP + VQA captioning within 77 tokens | Implemented, tested, used | `scripts/generate_captions.py`, `VLMCaptioner` |
| F4 | Aspect-bucket preparation and hold-out split | Implemented, tested, used | `scripts/prepare_dataset.py`, `prepare_dataset` |
| F5 | Packaging and the one-command Colab run | Implemented, used | `scripts/package_dataset.py`, `scripts/colab_train.sh` |
| F6 | LoRA training with checkpoint / resume | Implemented, tested (tiny model), run on T4 | `scripts/train_lora.py`, `LoraTrainer` |
| F7 | Text → painting generation | Implemented, tested, used | `scripts/generate.py`, `RaviVarmaGenerator.generate` |
| F8 | Prompt expansion (lexicon + optional LLM) | Implemented, tested; no measured gain | `PromptExpansionEngine` |
| F9 | ControlNet pose / edge control | Implemented, evaluated | `--pose-image`, `_load_controlnet_pipeline` |
| F10 | Face fix | Implemented, tested | `--face-fix`, `_fix_faces`, `detect_faces` |
| F11 | Hires fix | Implemented; GPU > 8 GB only | `--hires-scale`, `_redraw` |
| F12 | Photo → Varma | Implemented, run on T4 and Mac | `--init-image`, `_photo_redraw`, app tab |
| F13 | Paired CLIP evaluation | Implemented, tested, used | `scripts/evaluate.py`, `evaluate_run` |
| F14 | Pose adherence (PCK) evaluation | Implemented, tested, used | `scripts/eval_controlnet.py`, `pose_pck` |
| F15 | Gradio web app (5 tabs) | Implemented, browser-tested | `app/app.py` |
| F16 | Checkpoint Hub | Implemented | `list_checkpoints`, `loss_chart_svg` |
| F17 | Logging with secret redaction | Implemented, tested | `setup_logging`, `_RedactSecretsFilter` |
| F18 | Continuous integration | Implemented, passing | `.github/workflows/tests.yml` |
| F19 | SDXL training | **Not implemented** (scaffold only) | `configs/lora_sdxl.yaml` |

Each feature below is described with the same 18 points so they can be compared:
1. purpose
2. user value
3. entry point
4. inputs
5. outputs
6. main code
7. flow
8. data touched
9. configuration
10. dependencies
11. error handling
12. edge cases
13. performance
14. security / privacy
15. tests
16. status / evidence
17. limitations
18. improvements

---

## F1. Dataset download with provenance

| Point | Detail |
|---|---|
| 1 Purpose | Fetch public-domain Ravi Varma reproductions with their licence and source recorded |
| 2 User value | A legally clean, reproducible corpus; anyone can rebuild it with `--from-metadata` |
| 3 Entry | `python scripts/download_dataset.py --source commons|wellcome --limit N --min-side 800`; `make download` |
| 4 Inputs | Root Wikimedia category (`--category`), source, limit, minimum original side |
| 5 Outputs | `data/raw/NNNN.jpg`, one JSON line per file in `data/metadata/metadata.jsonl` (title, year, licence, source URL, original width / height) |
| 6 Main code | `api_get`, the category walk, `download_wellcome`, `next_free_index` |
| 7 Flow | Walk the category tree → query `imageinfo` (licence, size, 2048-px rendition URL) → skip known files and small originals → download → append metadata |
| 8 Data | `data/raw/`, `data/metadata/metadata.jsonl` |
| 9 Config | CLI flags only |
| 10 Dependencies | `requests`; Wikimedia Commons API; Wellcome catalogue + IIIF |
| 11 Errors | 429 / 5xx retried, honouring `Retry-After`; failed files are logged and skipped |
| 12 Edge cases | Re-runs never overwrite: `next_free_index` continues numbering, and already-recorded source URLs are skipped |
| 13 Performance | Sleeps about 2 s per file for rate limits; around 10 minutes per 300 files (**Inferred** from the sleep) |
| 14 Security | Public data only; a descriptive User-Agent; no credentials |
| 15 Tests | None directly (network); its outputs are covered by the validation tests |
| 16 Status | Used: 300 metadata records |
| 17 Limitations | The Commons category tree also contains non-paintings (stamps, photos); curation removes them (F2) |
| 18 Improvements | Cache API responses; add tests against a recorded HTTP fixture |

## F2. Validation, curation and de-duplication

| Point | Detail |
|---|---|
| 1 Purpose | Decide exactly which raw files may be trained on, with a reason for every exclusion |
| 2 User value | Garbage-in protection: no corrupt, upscaled, duplicated or non-painting images |
| 3 Entry | `python scripts/validate_dataset.py` (`--manifest` checks the prepared set); `make validate` |
| 4 Inputs | `data/raw/`, `metadata.jsonl`, `configs/dataset.yaml` (`validation`, `curation`) |
| 5 Outputs | `ValidationReport` (accepted, excluded with reasons, duplicate groups); `data/metadata/validation_report.json` |
| 6 Main code | `DatasetValidator.validate`, `perceptual_hash`, `safe_open_image`, `ValidationConfig`, `CurationConfig` |
| 7 Flow | Open each file safely → reject corrupt or < 384 px (checked against the **original** size) → curated exclusions → exact SHA-256 duplicates → curated duplicate groups → dHash near-duplicates (Hamming ≤ 40 of 256 bits) → keep the largest copy |
| 8 Data | Reads raw and metadata; writes the report only |
| 9 Config | `min_resolution`, `near_duplicate_threshold: 40`, `curation.exclude`, `curation.duplicate_groups`, `holdout` |
| 10 Dependencies | Pillow, NumPy |
| 11 Errors | Unreadable files become `DatasetIssue` entries; nothing crashes the run |
| 12 Edge cases | Decompression-bomb limit; files missing metadata fall back to the local size |
| 13 Performance | Near-duplicate search compares every pair, O(n²); fine for 300 files, slow for 30,000 |
| 14 Security | Pillow bomb protection (`safe_open_image`) |
| 15 Tests | `tests/test_curation.py` (13), `tests/test_dataset.py` (11) |
| 16 Status | 194 accepted (182 train + 12 hold-out); 26 upscaled copies caught |
| 17 Limitations | Perceptual hashing misses heavily cropped re-scans; those needed the curated groups |
| 18 Improvements | A BK-tree or LSH for near-duplicate search; CLIP-embedding duplicate suggestions |

## F3. Captioning (BLIP + VQA, 77-token budget)

| Point | Detail |
|---|---|
| 1 Purpose | Describe each painting's *content* so the trigger token learns only the *style* |
| 2 User value | Better prompt following and less leakage of frames or printed text into the style |
| 3 Entry | `python scripts/generate_captions.py`; `make captions`; `--reflatten` rebuilds flat captions without models |
| 4 Inputs | Accepted images, metadata titles, `caption_notes` in `configs/dataset.yaml` |
| 5 Outputs | `data/captions/NNNN.json` (sentence, VQA answers, palette); a flat caption in metadata |
| 6 Main code | `VLMCaptioner`, `assemble_caption`, `as_flat_caption`, `_short_title`, `measure_visual_properties`, `clip_token_counter` |
| 7 Flow | BLIP-large sentence → 12 BLIP-VQA questions (clothing, colour, jewellery, posture, action, expression, held object, setting, time of day…) → gate answers (drop "none", meta and anachronistic words) → add a measured palette, title and note → trim lowest-priority parts until ≤ 69 tokens (plus a 6-token trigger prefix) |
| 8 Data | `data/captions/`, `metadata.jsonl` |
| 9 Config | `--vlm-model`, `--vqa-model`, `caption_notes` |
| 10 Dependencies | Transformers BLIP models; the SD1.5 CLIP tokenizer |
| 11 Errors | Refuses to silently fall back to placeholder captions (a v2 bug, **Confirmed** in `docs/RESULTS.md`); `--no-vlm` is an explicit opt-in |
| 12 Edge cases | Long titles shortened; `--overwrite` needed to recaption |
| 13 Performance | About 1–2 s per image on MPS (**Inferred**) |
| 14 Security | Local models only |
| 15 Tests | `tests/test_captions.py` (10): token budget, filters, assembly |
| 16 Status | 194 sidecars |
| 17 Limitations | BLIP misreads mythological identities; the VQA answers are generic |
| 18 Improvements | A stronger VLM (e.g. Florence-2 / Qwen-VL); human review of the hold-out captions |

## F4. Aspect-bucket preparation and hold-out

| Point | Detail |
|---|---|
| 1 Purpose | Produce training-ready images without cropping heads and feet off tall canvases |
| 2 User value | Whole figures in training → whole figures in output |
| 3 Entry | `python scripts/prepare_dataset.py`; `make prepare` |
| 4 Inputs | The validator's accepted list and captions |
| 5 Outputs | `data/processed/*.png` + `manifest.jsonl` (182); `data/reference/` + `reference.jsonl` (12) |
| 6 Main code | `prepare_dataset`, `load_manifest`, `check_manifest`, border trimming in `utils/image.py` |
| 7 Flow | Trim scanner borders → pick the bucket nearest the aspect ratio with ~512² area (e.g. 448×640) → resize → write the image and a manifest row; hold-out files go to `data/reference/` |
| 8 Data | `data/processed/`, `data/reference/` |
| 9 Config | `target_resolution: 512`, `aspect_bucketing: true`, `holdout` list |
| 10 Dependencies | Pillow, NumPy |
| 11 Errors | `check_manifest` reports missing files, empty captions and over-long captions |
| 12 Edge cases | Extremely wide images snap to the most extreme bucket |
| 13 Performance | Seconds |
| 14 Security | n/a |
| 15 Tests | `tests/test_dataset.py` |
| 16 Status | 182 / 12 split in the repo |
| 17 Limitations | A fixed 12-image hold-out; evaluation variance is high |
| 18 Improvements | Stratify the hold-out by subject (portrait / mythology / group) |

## F5. Packaging and the Colab run

| Point | Detail |
|---|---|
| 1 Purpose | Move the prepared dataset to a free GPU and run the whole training + evaluation sequence with one command |
| 2 User value | Repeatable training without a local GPU |
| 3 Entry | `make package-data` → `dist/ravi_varma_dataset.zip`; in Colab `TAG=v3 bash scripts/colab_train.sh` |
| 4 Inputs | `data/processed/`, `data/reference/`, the repo |
| 5 Outputs | `checkpoints/lora/ravi_varma_<TAG>/`, sample images, evaluations, `ravi_varma_results.zip` |
| 6 Main code | `scripts/package_dataset.py`, `scripts/colab_train.sh`, `requirements-colab.txt` |
| 7 Flow | Install pinned packages → smoke test → full training → samples (with and without LoRA) → evaluation → hires samples → zip |
| 8 Data | Writes only under `outputs/` and `checkpoints/` |
| 9 Config | `TAG`, `configs/lora_sd15.yaml` |
| 10 Dependencies | A Colab T4 runtime (pinned to 2026.07) |
| 11 Errors | `set -e`; the smoke test fails fast before the hour-long run |
| 12 Edge cases | Re-running with the same `TAG` resumes from checkpoints |
| 13 Performance | About 58 min of training for 1500 steps on a T4 (**Measured**: 3492.8 s in `training_summary.json`) |
| 14 Security | No secrets needed; the dataset zip contains only public-domain images |
| 15 Tests | Not unit-tested (shell); exercised by real runs |
| 16 Status | Used for v2 and v3 |
| 17 Limitations | Colab sessions expire; manual upload and download |
| 18 Improvements | Push weights to the Hugging Face Hub from the notebook |

## F6. LoRA training

| Point | Detail |
|---|---|
| 1 Purpose | Teach SD1.5 the `<rvvarma>` style by training small low-rank adapters on the UNet attention layers |
| 2 User value | A 12.8 MB style file instead of a 4 GB model |
| 3 Entry | `python scripts/train_lora.py --config configs/lora_sd15.yaml [--smoke-test]`; `make train` |
| 4 Inputs | `manifest.jsonl`, processed images, config |
| 5 Outputs | `pytorch_lora_weights.safetensors`, `checkpoint-250…1500/`, `training_config.yaml`, `training_summary.json`, `loss_history.csv`, validation images |
| 6 Main code | `LoraTrainer` (`_load_models`, `_apply_lora`, `train`), `CaptionCollator`, `BucketBatchSampler`, `_min_snr_weights`, `CheckpointManager`, `ValidationImageCallback`, `LossLogger` |
| 7 Flow | See §3.5: encode → noise (+ offset) → UNet+LoRA predicts noise → Min-SNR-weighted MSE → update LoRA only |
| 8 Data | Reads the dataset; writes `checkpoints/` and `outputs/logs/` |
| 9 Config | rank 16, alpha 16, dropout 0.05, `to_q/k/v/out.0`, lr, 1500 steps, grad-accum 4, fp16, Min-SNR γ 5, noise offset 0.05, caption dropout 0.1 |
| 10 Dependencies | PyTorch, Diffusers, PEFT, Accelerate, CUDA |
| 11 Errors | `check_ml_stack_available`; SDXL raises `NotImplementedError`; `_apply_lora` asserts that only LoRA parameters are trainable |
| 12 Edge cases | Resume from the latest checkpoint; the warm-up counted in optimizer steps (a v2 fix) |
| 13 Performance | ~58 min on a T4; 3.2 M trainable parameters |
| 14 Security | Weights saved as safetensors (no pickle) |
| 15 Tests | `tests/test_training.py` (5) on the 1 MB tiny SD model: train + checkpoint export + resume, exported LoRA loads into a pipeline, only LoRA parameters trainable, v3 options run, caption dropout keeps only the trigger |
| 16 Status | v3 trained: 182 images, 1500 steps, mean loss of the last 50 steps 0.118 |
| 17 Limitations | The loss doesn't track style; no rank ablation |
| 18 Improvements | Rank 8 / 32 ablation; separate the data changes from the training changes |

## F7. Text → painting

| Point | Detail |
|---|---|
| 1 Purpose | Render a prompt in Varma's style |
| 2 User value | The core demo |
| 3 Entry | `scripts/generate.py --prompt …`; app **Generate** tab |
| 4 Inputs | Prompt, negative prompt, seed(s), steps, guidance, size, LoRA scale, LoRA path |
| 5 Outputs | `outputs/generated/<time>_<hash>.png` + `.json` sidecar |
| 6 Main code | `RaviVarmaGenerator.load / generate / _save`, `resolve_lora_weights`, `SCHEDULER_MAP` |
| 7 Flow | Load SD1.5 fp16 + VAE + LoRA once → add the trigger if missing → a CPU `torch.Generator(seed)` → run the pipeline → black-image guard → optional hires / face fix → save |
| 8 Data | Writes `outputs/generated/` |
| 9 Config | `configs/generation.yaml`: model, `lora_path` (v3), `vae`, scheduler `dpmsolver_multistep`, steps 30, guidance 7.5, **512×512** default size (the app defaults to 448×640 and the docs pass `--width 448 --height 640`), negative prompt, hardware options |
| 10 Dependencies | Diffusers, the HF Hub on first run |
| 11 Errors | `parser.error` for an empty prompt or missing files; a NaN / black image raises a clear `RuntimeError` |
| 12 Edge cases | Seed 0 in the app means random; `--no-lora` for baselines; `--skip-existing` for prompt suites |
| 13 Performance | ~1.5–2.5 min per image on an 8 GB Mac (139 s **Measured** in the README screenshot); ~5 s on a T4 |
| 14 Security | Local only |
| 15 Tests | `tests/test_generation.py` (24, using the tiny model) |
| 16 Status | Used for all results |
| 17 Limitations | Hands, crowds, complex actions |
| 18 Improvements | Progress callback in the app; make 448×640 the config default to match the app |

## F8. Prompt expansion (Milestone 3)

| Point | Detail |
|---|---|
| 1 Purpose | Turn a story sentence into a prompt with each character's standard iconography, within 75 tokens |
| 2 User value | Users needn't know that Krishna is "blue-skinned with a peacock-feather crown" |
| 3 Entry | `--expand` (`--dry-run` prints only); the app's "Expand" checkbox |
| 4 Inputs | Scene text; `configs/epic_lexicon.yaml` (27 characters, 18 places) |
| 5 Outputs | `ExpandedPrompt` (trigger, scene, looks, environment, style cues) → `final_prompt()` |
| 6 Main code | `LocalPromptEngine`, `LLMPromptEngine`, `PromptExpansionEngine`, `_find_characters`, `_extract_environment` |
| 7 Flow | Find characters by name or alias → add looks and setting → optional LLM adds short phrases (JSON mode) → assemble by priority, dropping the lowest first until ≤ 75 tokens |
| 8 Data | Reads the lexicon |
| 9 Config | `.env`: `PROMPT_EXPANSION_PROVIDER`, `OPENAI_BASE_URL`, model name, key |
| 10 Dependencies | Optional `openai` / `anthropic` SDK |
| 11 Errors | Any LLM failure falls back to the local engine |
| 12 Edge cases | The user's scene is always kept verbatim; the negative prompt is never replaced |
| 13 Performance | Local: milliseconds. LLM: about 1 s network |
| 14 Security | Only scene text leaves the machine; keys read from env and never logged |
| 15 Tests | `tests/test_prompt_engine.py` (18) |
| 16 Status | **Measured:** CLIP raw 0.326, local 0.317, LLM 0.320 on 4 scenes: no gain |
| 17 Limitations | Cannot fix complex actions |
| 18 Improvements | Regional prompting per character; a larger evaluation set |

## F9. ControlNet pose / edge control (Milestone 4)

| Point | Detail |
|---|---|
| 1 Purpose | Fix *where* figures are while the LoRA decides *how they look* |
| 2 User value | Correct figure count and pose in multi-figure scenes |
| 3 Entry | `--pose-image ref.jpg --controlnet-type openpose|canny`; the app's pose reference box |
| 4 Inputs | Reference image, type, strength |
| 5 Outputs | Image + `*_control.png` (skeleton or edge map) |
| 6 Main code | `make_conditioning_image`, `make_canny_conditioning`, `OpenPoseDetector`, `_load_controlnet_pipeline` |
| 7 Flow | Crop the reference to the output aspect → extract the skeleton / edges → build a ControlNet pipeline from shared components → generate |
| 8 Data | Writes the control map next to the output |
| 9 Config | `controlnet:` models and default strength in `generation.yaml` |
| 10 Dependencies | `controlnet-aux`, OpenCV, the lllyasviel ControlNet weights |
| 11 Errors | No person detected → an empty skeleton map, and the image is still generated (**Inferred** from the code path; there is no explicit warning) |
| 12 Edge cases | Aspect mismatch handled by cropping |
| 13 Performance | +~1.4 GB memory; ~1.5–2 min on the Mac |
| 14 Security | Local |
| 15 Tests | `tests/test_generation.py`: Canny map keeps the size and is greyscale-as-RGB; no test loads a real ControlNet (too large for CI) |
| 16 Status | **Measured:** PCK 0.17 → 0.86; style 0.704 → 0.692 |
| 17 Limitations | OpenPose fails on crowded paintings; hands are not controlled |
| 18 Improvements | DWPose; multi-ControlNet (pose + depth) |

## F10. Face fix

| Point | Detail |
|---|---|
| 1 Purpose | Sharpen small faces (≈40 px at 448×640) that the VAE cannot render cleanly |
| 2 User value | Clear eyes and lips without a bigger model |
| 3 Entry | `--face-fix`; on by default in the app |
| 4 Inputs | The generated image, the prompt |
| 5 Outputs | The image with redrawn faces; face boxes in the metadata |
| 6 Main code | `detect_faces` (Haar + eye check), `_fix_faces`, `_redraw` |
| 7 Flow | Detect faces → keep boxes with an eye inside → crop with margin → img2img at 512 px, strength 0.4, prompt + "detailed face" → paste back with a feathered oval mask |
| 8 Data | In memory |
| 9 Config | `generation.face_fix` (pydantic default `false`, not set in `generation.yaml`); `--face-fix` turns it on; the app's checkbox is on by default |
| 10 Dependencies | OpenCV Haar cascades |
| 11 Errors | No faces → unchanged image |
| 12 Edge cases | The eye check prevents redrawing medals or cloth folds as faces |
| 13 Performance | +~20 s per face on the Mac |
| 14 Security | n/a |
| 15 Tests | `test_face_fix_redraws_and_blends_only_the_face` (stubbed detector and redraw) |
| 16 Status | Visual evidence: `outputs/evaluations/face_fix_compare.jpg` (not measured) |
| 17 Limitations | Misses profiles and small faces in crowds |
| 18 Improvements | A DNN face detector; measure with a face-quality metric |

## F11. Hires fix

| Point | Detail |
|---|---|
| 1 Purpose | Higher resolution and detail: upscale ×1.5, then redraw at strength 0.4 |
| 2 User value | Sharper texture for presentation images |
| 3 Entry | `--hires-scale 1.5` |
| 4–5 I/O | Generated image → 672×960 image |
| 6 Main code | `generate(hires_scale=…)`, `_redraw` |
| 7 Flow | Lanczos upscale → img2img over the whole image with the same prompt and seed |
| 8–10 | In memory; `generation.hires_scale` (default 1.0 = off); no extra dependency |
| 11 Errors | Out-of-memory on 8 GB (documented, not caught) |
| 12 Edge cases | Sizes rounded to multiples of 8 |
| 13 Performance | ~2.3× the pixels; needs > 8 GB |
| 14 Security | n/a |
| 15 Tests | No dedicated test |
| 16 Status | Run on the T4 (`outputs/evaluations/v2_v3_hires_compare.jpg`) |
| 17 Limitations | Not usable on the target Mac |
| 18 Improvements | Tiled VAE decoding to fit 8 GB |

## F12. Photo → Varma

| Point | Detail |
|---|---|
| 1 Purpose | Repaint a user's photo in Varma's style while keeping its composition and likeness |
| 2 User value | The most intuitive demo |
| 3 Entry | App **Photo → Varma** tab; `--init-image photo.jpg --strength 0.55` |
| 4 Inputs | Photo, optional description, strength |
| 5 Outputs | Painting + edge map (`*_control.png`); the photo itself is **not** saved |
| 6 Main code | `_photo_redraw`, `_load_controlnet_pipeline("canny", img2img=True)`, `run_photo_to_painting` |
| 7 Flow | Crop to 2:1 if more extreme → resize to the ~448×640 area → Canny edges → ControlNet-img2img with a neutral prompt → face fix |
| 8 Data | Writes outputs only |
| 9 Config | Strength default 0.55 |
| 10 Dependencies | The Canny ControlNet |
| 11 Errors | Missing photo → a UI warning |
| 12 Edge cases | Facial-hair terms are stripped from the negative prompt so a real moustache survives |
| 13 Performance | ~15 s on a T4; on the Mac, needs other apps closed |
| 14 Security | Processed locally; no upload anywhere |
| 15 Tests | `test_photo_to_painting_uses_img2img_at_training_size`, `test_photo_mode_keeps_facial_hair_out_of_the_negative_prompt`, `test_photo_of_any_shape_gets_a_bounded_size` |
| 16 Status | `outputs/evaluations/photo_to_varma.jpg` |
| 17 Limitations | Masks and modern costumes get reinterpreted; small round objects can become faces at high strength |
| 18 Improvements | IP-Adapter for identity; a depth ControlNet |

## F13. Paired CLIP evaluation

| Point | Detail |
|---|---|
| 1 Purpose | Measure the style shift against the base model honestly |
| 2 User value | Numbers that can be defended |
| 3 Entry | `python scripts/evaluate.py --input … --baseline …`; `make evaluate` |
| 4 Inputs | Generated images + sidecars, base images, `data/reference/`, training manifest |
| 5 Outputs | `summary.json`, per-image CSV, contact sheet in `outputs/evaluations/<run>/` |
| 6 Main code | `evaluate_run`, `_score_directory`, `_paired`, `text_image_similarity`, `embed_images` |
| 7 Flow | Embed references and training images once → per image: prompt adherence (trigger stripped), mean / top-3 style, nearest training similarity → diversity per prompt → paired deltas |
| 8 Data | Read-only, except the outputs |
| 9 Config | CLI flags |
| 10 Dependencies | open_clip ViT-B/32 |
| 11 Errors | CLIP missing → a clear message; affected tests are skipped |
| 12 Edge cases | Images without a sidecar → no prompt-adherence score |
| 13 Performance | Seconds on CPU for 30 images |
| 14 Security | n/a |
| 15 Tests | `tests/test_evaluation.py` (14) |
| 16 Status | **Measured:** style 0.669 → 0.701 (12/15 pairs), top-3 0.734 → 0.770, adherence 0.328 → 0.330, diversity 0.106 → 0.092, max training similarity 0.898 |
| 17 Limitations | CLIP is a coarse style judge; only 15 pairs |
| 18 Improvements | A human study; Gram-matrix or CSD style metrics; more seeds |

## F14. Pose adherence (PCK)

| Point | Detail |
|---|---|
| 1 Purpose | Measure whether ControlNet output matches the reference pose |
| 3 Entry | `python scripts/eval_controlnet.py render|report` |
| 5 Outputs | `outputs/evaluations/m4_controlnet/summary.json`, `grid.jpg` |
| 6 Main code | `pose_keypoints`, `_person_pck`, `pose_pck` |
| 7 Flow | OpenPose on the reference and the output → match people → share of reference joints within 0.1 of the image size |
| 11 / 12 | No person in the reference → `None` (nothing to compare); a reference person with no generated match → 0 |
| 15 Tests | `test_identical_pose_scores_one_and_missing_person_zero`, `test_partial_match_and_best_person_chosen` |
| 16 Status | **Measured:** LoRA 0.17, OpenPose 0.86, Canny 0.86 (6 references, seed 42) |
| 17 Limitations | It depends on OpenPose reading paintings; small sample |
| 2, 4, 8–10, 13, 14, 18 | Same as F9 and F13 |

## F15. Gradio web app

| Point | Detail |
|---|---|
| 1 Purpose | A demo UI over the same generator |
| 3 Entry | `python app/app.py [--server-port 7860] [--share]` → http://127.0.0.1:7860 |
| 4 / 5 | Tabs: Generate, Photo → Varma, Checkpoint Hub, Evaluation, About |
| 6 Main code | `build_app`, `_theme`, `run_generation`, `run_photo_to_painting`, `recent_images`, `app/styles.css` |
| 7 Flow | §3.9 |
| 11 Errors | `_require_gradio` gives a friendly message; `RuntimeError` is shown as a warning |
| 13 Performance | One model in memory; requests run one at a time |
| 14 Security | Bound to localhost; `--share` makes it public, so use it deliberately |
| 15 Tests | `test_checkpoint_hub_lists_steps_with_mean_loss`; the UI itself was browser-tested by hand (screenshots in `docs/screenshots/`), with no automated UI test |
| 17 Limitations | No progress bar; single user |
| 18 Improvements | A progress callback; a Hugging Face Space |

## F16. Checkpoint Hub

| Point | Detail |
|---|---|
| 1 Purpose | Compare checkpoints 250–1500 and switch the LoRA live |
| 6 Main code | `list_checkpoints`, `checkpoint_table_html`, `loss_chart_svg`, `RaviVarmaGenerator.set_lora` |
| 7 Flow | Scan `checkpoints/lora/*/checkpoint-*` → table of step, mean loss, size → inline SVG loss chart → the dropdown feeds `set_lora` |
| 17 Limitations | Only the final v3 weights ship with the repo; intermediate checkpoints exist only where training ran |

## F17. Logging with secret redaction

| Point | Detail |
|---|---|
| 1 Purpose | Consistent logs; never write tokens to logs |
| 6 Main code | `setup_logging`, `get_logger`, `_RedactSecretsFilter` |
| 7 Flow | Any message containing a sensitive key fragment (token, api_key, …) next to `=` or `:` is replaced wholesale |
| 15 Tests | `tests/test_logging.py` (4) |
| 17 Limitations | Best-effort and pattern-based; a bare secret without a key name is not caught |

## F18. Continuous integration

| Point | Detail |
|---|---|
| 1 Purpose | Every push runs lint and tests on a clean Ubuntu machine |
| 6 Main code | `.github/workflows/tests.yml` |
| 7 Flow | CPU torch → requirements → pyflakes → pre-fetch the tiny SD model and SD1.5 tokenizer → `pytest -q -rs` |
| 16 Status | Passing; 119 passed, 2 skipped (CLIP weights cached) |
| 17 Limitations | No GPU tests; nothing checks real image quality |

## F19. SDXL training (not implemented)

`configs/lora_sdxl.yaml` exists, but `LoraTrainer._load_models` raises `NotImplementedError` for SDXL: scaffolding
from the proposal, **not a feature**. It would need two text encoders, time-id conditioning and a ≥ 24 GB GPU (D-01).
