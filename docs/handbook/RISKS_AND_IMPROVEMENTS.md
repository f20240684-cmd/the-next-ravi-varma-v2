<!-- Extracted from the technical handbook (docs/handbook/src/); edit the source files and rerun build_handbook.py. Commit e2cae62. -->

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

## 11.3 Ranked improvements (impact ÷ effort)

| Rank | Improvement | Expected impact | Effort | Evidence basis |
|---|---|---|---|---|
| 1 | Human evaluation study | High (credibility) | Low | The main gap named in README / DECISIONS |
| 2 | Config hygiene: `extra="forbid"`, one negative prompt, 448×640 default | Medium | Very low | §6.6 |
| 3 | Fix the stale docs (Appendix A) | Medium (reviewer trust) | Very low | Appendix A |
| 4 | Ablation runs (rank, prints, data vs training) | High (understanding) | Medium (GPU hours) | D-03, D-08, D-16 confounds |
| 5 | Hugging Face Space + weights on the Hub | High (reach) | Medium | Not deployed now |
| 6 | DNN face detector (e.g. RetinaFace / MediaPipe) | Medium (faces in profile / crowds) | Low–medium | Known Haar misses |
| 7 | Tiled VAE to allow hires on 8 GB | Medium | Medium | Hires is GPU-only now |
| 8 | Stronger captioner (Florence-2 / Qwen-VL) for v4 | Medium | Medium | BLIP can't name deities |
| 9 | SDXL port | High (quality) | High | D-01 |
| 10 | Regional prompting / multi-ControlNet for multi-figure actions | Medium | High | M3 failure on complex actions |

# Appendix A. Inconsistencies, assumptions, limitations and open questions

## A.1 Documentation that disagrees with the code

Found while writing this handbook. Nothing was changed (documentation-only task). The code is taken as the truth.

| # | Where | Says | Reality | Suggested fix |
|---|---|---|---|---|
| 1 | `checkpoints/README.md` | Weights are "**not** committed to git" | The v3 final weights are committed (commit `f49f890`, `.gitignore` exception) | Update the paragraph |
| 2 | `src/ravi_varma/evaluation/run_eval.py` docstring | "evaluation framework planned for Milestone 4" | Milestone 4 is implemented | Reword |
| 3 | `configs/generation.yaml` comment | ControlNet "Milestone 4 (not yet implemented end-to-end)" | Implemented and evaluated; used whenever a pose image is given (`enabled: false` only means "no control by default") | Reword |
| 4 | `configs/lora_sdxl.yaml` | A full SDXL config | `NotImplementedError` in the trainer | Header comment or delete |
| 5 | `DEFAULT_NEGATIVE_PROMPT` (`prompt_engine.py`) | Its own negative prompt | Differs from `generation.yaml`'s (no facial-hair terms) | Single source |
| 6 | `ImageEvaluation.ssim / lpips` (`report.py`) | Metric fields | Never computed (always `None`) | Remove or implement |
| 7 | `IMPLEMENTATION_MANUAL.md`, `notebooks/03_lora_training.ipynb` | Older workflow | `scripts/colab_train.sh` is the maintained path (the README says so for the notebook) | Mark as historical |
| 8 | `docs/screenshots/05_about.png` | About tab | Captured before the About text was corrected in commit `e2cae62` | Re-capture |
| 9 | `configs/generation.yaml` default size | 512×512 | App and docs use 448×640 | Align |
| 10 | Checkpoint Hub image caption | Progression image labelled "(v2 run)" while the table shows v3 | Correctly labelled, but mixes runs | Render a v3 progression |

## A.2 Assumptions made in this handbook

- **Timings** for downloading and captioning are estimates from sleep intervals and model sizes (marked **Inferred**).
- **Model download sizes** are approximate public sizes, not measured on this machine.
- **Rationales in `docs/DECISIONS.md`** are treated as the team's recorded reasoning, although that file was written after the fact.
- **The evaluator** has internet on first run and Python 3.11.

## A.3 Limitations (consolidated)

1. SD1.5 resolution and anatomy (hands, crowds).
2. Complex actions are not rendered.
3. Small evaluation samples; CLIP is a coarse style judge; no human study.
4. v3 data and training changes are confounded.
5. The effect of the prints is untested.
6. Diversity dropped in v3.
7. Hires fix is GPU-only; the Mac needs free memory for ControlNet.
8. Haar face detection misses profiles.
9. Single-user local app, no authentication. `--share` is public.
10. No safety checker.

## A.4 Open questions

- Would rank 8 or 32 give a better style / diversity trade-off?
- Do the 29 chromolithograph prints help or flatten the style?
- Which v3 change (data vs Min-SNR / noise offset / dropout) produced the gain?
- Would humans prefer v3 over base in a blind test, and at what rate?
- Does the face fix measurably improve faces (e.g. with a face-quality model), or only visually?
- Is the LLM expansion useful on a larger, action-free evaluation set?
