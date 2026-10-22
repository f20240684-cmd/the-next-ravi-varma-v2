# Appendix B. Verification report

How this handbook and its PDF were checked. The checks are repeatable: the build script prints the diagram and
overflow statistics, and the PDF checks used PyMuPDF.

## B.1 Scope check: documentation only

| Check | Method | Result |
|---|---|---|
| No application code, configuration, dependency or data changed | `git status --short` and `git diff --stat` after the build | Only the new `docs/handbook/` folder is added; `git diff` (tracked files) is empty |
| No existing file overwritten | All outputs are written inside `docs/handbook/` | Confirmed |
| No secrets included | Only placeholders (`<your-key-here>`) and variable names appear; `.env` was never read | Confirmed |

## B.2 Content accuracy checks

Every factual statement was checked against the repository at commit `e2cae62`:
- **File paths, symbols and CLI flags:** `grep` over `src/`, `scripts/`, `app/`.
- **Numbers:** read from these files:
  - `checkpoints/lora/ravi_varma_v3/training_summary.json`
  - `outputs/evaluations/m4_controlnet/summary.json`
  - `README.md` and `docs/RESULTS.md` (which cite `outputs/evaluations/lora_v3_vs_base/summary.json`)
  - `data/*` record counts
- **Test claims:** matched to test function names.

Draft statements that the checks **corrected** before publication:

| Draft claim | Corrected to | Evidence |
|---|---|---|
| Default generation size 448×640 | Config default is **512×512**; the app and docs use 448×640 | `configs/generation.yaml`, `GenerationParamsConfig` |
| Face fix has no config key | `generation.face_fix` exists (default `false`) | `config.py` |
| Training tests cover "Min-SNR maths" | They cover train/export/resume, reload, LoRA-only params, v3 options, caption dropout | `tests/test_training.py` |
| PCK with no person is 0 | `None` when the reference has no person; 0 for an unmatched person | `pose_score.pose_pck` |
| Photo tab shows the edge map | The edge map is saved, not shown; the outputs are image, details, warning | `app/app.py` wiring |
| Checkpoint Hub loads on tab open | Built once when the app starts | `app/app.py` |
| Safety checker absence "inferred" | Confirmed: `safety_checker=None` | `pipeline.py`, `lora_trainer.py` |
| Demo chip "Saraswati" | The chips are Damayanti, Krishna and Arjuna, Shakuntala | `app/components.py` `EXAMPLES` |
| VAE download ~160 MB | ~330 MB | Public model size (approximate) |

## B.3 PDF checks

| Check | Method | Result |
|---|---|---|
| PDF exists and opens | PyMuPDF `open` | `docs/handbook/Project_Complete_Technical_Handbook.pdf`, **77 pages**, A4 |
| Text is extractable (not images) | `page.get_text()` on every page | **about 135,000 characters** extracted |
| Diagrams render | Mermaid 10.9.1 in headless Chromium; each diagram rendered separately with errors captured | **17 / 17 rendered, 0 errors** |
| Diagrams readable | Rendered ÷ natural width per diagram | All ≥ **0.59**; the wide flowcharts were redrawn top-to-bottom after a first build showed them at 0.39 |
| Nothing overflows the page | Tables, code blocks and SVGs wider than the body | **0** (a module table overflowed once; fixed with soft breaks in long identifiers) |
| Page numbers | Footer "Page X of Y" searched on every page | Present on all pages |
| Table of contents | Each of the **113** entries (chapters + sections) checked for its heading text on the stated page | **113 / 113 correct** (a first version matched a chapter title inside a table; chapters are now matched at the page top) |
| Visual inspection | Pages rendered to PNG and inspected (cover, contents, architecture, features, flows, appendix) | Layout, tables and diagrams legible |

## B.4 What was not verified

- **Rendering in other PDF viewers.** Only checked with PyMuPDF and the Chromium print engine.
- **Model download sizes and some timings** are approximate and labelled **Inferred** in the text.
- **The rationales in `docs/DECISIONS.md`** are the team's recorded reasoning; their historical accuracy cannot be checked from the code.
- **No code was executed** to re-measure the results. The numbers are quoted from committed result files.

## B.5 How to rebuild

```bash
source .venv/bin/activate
pip install pymupdf          # optional: page numbers in the contents
python docs/handbook/build_handbook.py
```

The script writes `Project_Complete_Technical_Handbook.md`, the extracted deliverables and the PDF, then prints the
diagram and overflow statistics.
