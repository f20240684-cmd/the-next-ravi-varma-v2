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
