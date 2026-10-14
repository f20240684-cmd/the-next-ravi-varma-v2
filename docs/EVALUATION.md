# Evaluation protocol (Milestones 2–4)

This is a deliberately **simple** evaluation: enough to tell whether a trained
LoRA checkpoint is better than the base model and which checkpoint/strength to
use. The ControlNet (pose) evaluation added at Milestone 4 is described at the end.

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

## Milestone 4: pose adherence with ControlNet

`python scripts/eval_controlnet.py render --variant {lora,openpose,canny}` then `... report`.

- **Pose references.** 6 of the 12 held-out paintings (0225, 0189, 0212, 0236, 0026, 0173): standing, seated,
  reclining, a male portrait, two figures, and a mother with child. They were never seen in training.
- **Rendering.** Each painting's own caption and seed 42, three ways: LoRA only, LoRA + OpenPose, LoRA + Canny.
  The reference is cropped to the output size exactly as the generator crops it.
- **Pose adherence (PCK, `evaluation/pose_score.py`).** OpenPose finds up to 18 body joints per person in the
  reference and in the output. Each reference person is matched to the output person who best fits, and PCK is the
  share of that reference person's visible joints within 0.1 (coordinates normalised to width and height). We
  report the mean over reference people. A missing person scores 0.
- **Style.** CLIP similarity to the held-out paintings **excluding the pose source**. Otherwise Canny, which traces
  its source, would be rewarded for copying.
- **Prompt adherence.** CLIP similarity to the caption, with the trigger removed.

Results (`outputs/evaluations/m4_controlnet/summary.json`): PCK 0.17 (LoRA only) → **0.86** (OpenPose) / 0.86
(Canny); style 0.704 / 0.692 / 0.714; prompt adherence 0.324 / 0.339 / 0.352.

Limitations:
- PCK uses OpenPose to judge OpenPose-guided images, so it can share the detector's blind spots.
- It ignores hands and faces.
- It needs people that OpenPose can detect in a *painting*.
- With 6 images × 1 seed, treat differences under ~0.05 as noise.
