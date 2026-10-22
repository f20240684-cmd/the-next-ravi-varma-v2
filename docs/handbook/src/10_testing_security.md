# 10. Testing, reliability and security audit

## 10.1 What the tests prove

**121 tests**: 99 test functions, some parametrised (`pytest --collect-only`). On CI (Ubuntu, CPU) 119 pass and 2 are
skipped; locally all can run when CLIP weights are cached.

| File | Tests | What it proves |
|---|---|---|
| `test_curation.py` | 13 (+ parametrised) | Accepted / excluded sets, auditable report JSON, hold-out, stale processed files cleared, manifest check (placeholder captions, missing trigger), bucket sizes, the 448×640 mapping, bucket sampler, border trimming, title / year cleaning, metadata round-trip |
| `test_dataset.py` | 11 | Low-resolution and corrupt images flagged, empty dataset, missing / invalid metadata, preparation from captions, manifest round-trip, downloader numbering after the highest existing file |
| `test_captions.py` | 10 (+ parametrised) | Priority order and budget trimming, sidecar round-trip, meta / anachronistic answers dropped, period rewrites, heuristic captioner opt-in only, note kept inside the budget, title shortening |
| `test_training.py` | 5 | **A real LoRA training run** on the 1 MB tiny SD: checkpoint, resume, export, reload; only LoRA trainable; v3 options; caption dropout |
| `test_generation.py` | 24 (+ parametrised) | Config validation, Canny maps, seed recording, metadata keys, LoRA path resolution, missing-LoRA errors, ML-stack-missing message, trigger insertion, the sample suite, MPS slicing rule, negative prompt ≤ 77 tokens, Checkpoint Hub, face-fix blend, photo sizing and negative-prompt stripping |
| `test_prompt_engine.py` | 18 | Trigger on / off, verbatim scene, empty scene rejected, no invented environment, characters in scene order, aliases, budget trimming, default vs custom negative prompt, LLM fallback when unconfigured or raising (mocked), LLM output grounded in the lexicon |
| `test_evaluation.py` | 14 | CLIP unavailable paths, sidecar prompts, paired metrics, the review sheet is never overwritten, trigger stripping, PCK cases |
| `test_logging.py` | 4 | Secret redaction |

**Not proven by tests:**
- fp16 CUDA training (proven by the Colab runs)
- real image quality (CLIP evaluation and eyes)
- real ControlNet / BLIP / OpenPose model loading (too large for CI)
- the downloader against live APIs
- the Gradio layout

## 10.2 Reliability review

| Area | Strength | Weakness |
|---|---|---|
| Reproducibility | Seeds via a CPU generator; sidecars; pinned versions; committed metadata and captions; shipped weights | Raw images must be downloaded again (and Commons can change or delete files) |
| Failure visibility | Explicit errors (missing LoRA, black image, placeholder captions) | YAML typos are silently ignored (pydantic default `extra="ignore"`) |
| Resource limits | Shared components; `empty_cache`; photo sizes bounded (`test_photo_of_any_shape_gets_a_bounded_size`) | No memory pre-check; hires can OOM without a friendly message |
| Concurrency | Gradio processes one event at a time per handler | One shared generator: concurrent users would queue; `set_lora` changes state for everyone |
| Long jobs | Checkpoint/resume every 250 steps | The app has no progress indicator or cancel |
| Dependencies | Pins; a CI on a clean machine | Older Diffusers/PEFT; upgrades are untested |

## 10.3 Security and privacy review

| Topic | Finding | Severity |
|---|---|---|
| Secrets | `.env` is git-ignored; `.env.example` has empty values; history was scanned for key patterns during the audit (commit `dd26570`); logs redact key-like messages | Low risk |
| Network exposure | The app binds to `127.0.0.1` by default. `--share` creates a public Gradio URL with **no authentication**, which would expose the GPU and the filesystem output folder to anyone with the link | Medium if misused |
| Uploaded photos | Processed in memory; not stored; not sent anywhere. The edge map *is* saved and could reveal the photo's outlines | Low |
| Third-party data flow | Only the scene text goes to the LLM provider, only when it is enabled | Low |
| Model loading | safetensors (no pickle) for LoRA; Hub models from known publishers; no `trust_remote_code` anywhere in `src/` | Low |
| Malicious images | Pillow's decompression-bomb guard; aspect clamp 2:1 | Low |
| Path inputs | The Evaluation tab accepts a free-text reference directory: a local user can point it at any folder they can read. Acceptable for a local tool, not for a hosted one | Low (local) / Medium (if hosted) |
| Content | No safety checker is used (`safety_checker=None` in `RaviVarmaGenerator.load` and the trainer's validation pipeline). Outputs are labelled AI-generated | Medium for public hosting |
| Licences | Training images are public domain / CC BY with recorded provenance. SD1.5 is under CreativeML OpenRAIL-M, which carries use restrictions that apply to outputs and redistribution | Note |

## 10.4 Prioritised risks and improvements

| # | Priority | Item | Why | Effort |
|---|---|---|---|---|
| 1 | High | Add `model_config = ConfigDict(extra="forbid")` to the config models | Typos silently fall back to defaults, which could change results unnoticed | 30 min + test |
| 2 | High | Remove the stale statements (Appendix A): `checkpoints/README.md`, the `run_eval.py` docstring, the ControlNet comment in `generation.yaml`, `IMPLEMENTATION_MANUAL.md` / notebook 03 | Reviewers read these first | 1 h |
| 3 | High | Run a blind human evaluation (base vs v3, real vs generated) | CLIP is a weak style judge; it's the main evidence gap | 1 day |
| 4 | Medium | One style-similarity implementation, used everywhere | Three copies can drift | 2 h |
| 5 | Medium | One negative-prompt source (config), deleting or deriving `DEFAULT_NEGATIVE_PROMPT` | Two sources of truth | 30 min |
| 6 | Medium | Warn and refuse `--share` unless a username / password is set (`auth=`) | Public exposure | 30 min |
| 7 | Medium | Ablations: rank 8/16/32; with and without prints; data vs training changes | Unknown causes of the v3 gain | 3–4 GPU-hours |
| 8 | Medium | Make the config default size 448×640 to match the app and docs | CLI users get 512×512 unexpectedly | 5 min |
| 9 | Low | Remove the unused SSIM / LPIPS fields, or implement them | Misleading empty fields | 15 min |
| 10 | Low | Replace the O(n²) near-duplicate search (BK-tree) | Only matters at scale | 2 h |
| 11 | Low | Progress bar and memory pre-check in the app | UX on 8 GB machines | 2 h |
| 12 | Low | Delete `configs/lora_sdxl.yaml`, or label it "not implemented" in its header | Avoid the impression of SDXL support | 5 min |
