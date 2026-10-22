<!-- Extracted from the technical handbook (docs/handbook/src/); edit the source files and rerun build_handbook.py. Commit e2cae62. -->

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
