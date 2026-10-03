# Evaluating the style LoRA (Milestone 2 scope)

This is a deliberately **simple** evaluation: enough to tell whether a trained
LoRA checkpoint is better than the base model and which checkpoint/strength to
use. The fuller multimodal evaluation framework (ControlNet-aware metrics,
larger human study) is Milestone 4 work.

## Protocol

1. Render the fixed prompt suite (`configs/prompts/sample_prompts.yaml`, 5 prompts)
   with **the same seeds** twice: once with the LoRA, once with the base model.

   ```bash
   python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44 \
       --output-dir outputs/generated/lora
   python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44 \
       --no-lora --output-dir outputs/generated/baseline
   ```

2. Score both sets and compare them pairwise (same prompt id + seed):

   ```bash
   python scripts/evaluate.py --input outputs/generated/lora --baseline outputs/generated/baseline \
       --output-dir outputs/evaluations/lora_vs_base
   ```

3. Open `contact_sheet.jpg` / `contact_sheet_baseline.jpg` and fill in
   `qualitative_review.csv` (1–5 per criterion, rubric below). The file is never
   overwritten by later runs.

4. For checkpoint selection, repeat steps 1–2 with
   `--lora-path checkpoints/lora/ravi_varma/checkpoint-<step>` and/or `--lora-scale 0.6 / 0.8`.

## What each criterion uses

| Criterion | Automatic signal | Human rubric column |
|---|---|---|
| Prompt adherence | `prompt_adherence`: CLIP ViT-B/32 image–text cosine, trigger token removed | `prompt_adherence_1to5` |
| Style consistency | `style_ref_mean`, `style_ref_top3`: CLIP image–image cosine to the **12 held-out** paintings in `data/reference/` | `style_consistency_1to5` |
| Diversity | `diversity`: mean pairwise CLIP distance between seeds of the same prompt | — |
| Memorisation | `nearest_train_sim`: similarity to the closest *training* image; > 0.95 is flagged | — |
| Visual quality | none (no reliable automatic metric for paintings) | `visual_quality_1to5` |
| Anatomy | none (faces, eyes, hands, limb count need a human) | `anatomy_1to5` |

### Rubric (1 = poor, 5 = excellent)

- **visual_quality** — coherent image, no smearing/noise/artifacts, finished-painting look.
- **anatomy** — faces symmetric with plausible eyes; hands with five fingers; correct number of limbs; no duplicated or merged people.
- **style_consistency** — would sit plausibly next to the reference paintings: academic oil rendering, soft modelling of skin, rich textiles, Varma's palette and staging.
- **prompt_adherence** — the requested subjects, setting and clothing are present.

## Limitations (read before quoting any number)

- **Absolute CLIP scores are not meaningful on their own.** ViT-B/32 image–text
  similarity typically sits around 0.20–0.35 for any reasonable image; only the
  *paired difference* against the base model on identical prompts and seeds says
  something.
- **Style similarity mixes content and style.** A palace scene is closer to palace
  references whatever its painting technique. CLIP was not trained to separate
  artistic technique from subject matter, so the metric is a coarse proxy.
- **Twelve reference paintings is a small sample**, chosen for coverage, so the mean
  is noisy; `style_ref_top3` is less diluted by unrelated subjects.
- **CLIP is blind to anatomy.** An image with six fingers or a melted face can score
  well, which is why anatomy is rated by a person.
- **Diversity can be high for bad reasons** (incoherent images differ a lot) — read it
  together with the visual-quality ratings.
- **The memorisation check only catches near-copies** of whole training images, not
  copied fragments.
- No metric here measures "authenticity". The outputs are Ravi Varma–*inspired*
  generations and must be presented as such.
