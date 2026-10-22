# 8. First principles

This chapter explains the ideas the code relies on, from the ground up, using this project's numbers. Formulas are
written in plain notation.

## 8.1 What a diffusion model learns

**Forward process (fixed, no learning).** Take an image x₀ and add Gaussian noise in T = 1000 small steps until
nothing is left but noise. At any step t there's a closed form:

  xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε,  with ε ~ N(0, I)

ᾱₜ goes from ≈1 (clean) to ≈0 (pure noise) along a fixed schedule.

**What the network learns.** Given the noisy xₜ, the step t and a text condition c, predict the noise ε that was
added. The training loss is simply:

  L = ‖ ε − ε_θ(xₜ, t, c) ‖²

That's the `F.mse_loss(model_pred, target)` in `LoraTrainer.train`.

**Generation (sampling).** Start from pure noise and repeatedly subtract a bit of predicted noise. A *scheduler*
decides the step sizes. This project uses DPM-Solver multistep (`scheduler: dpmsolver_multistep`), which reaches good
images in about 30 steps instead of 1000.

**Why loss doesn't show style.** Each training step picks a random t. Predicting noise at high t is easy and at low t
is hard, so the per-step loss jumps around a flat mean (0.17–0.18 for v2) whatever the style quality. That's why
checkpoints are judged by images, not by loss (the Checkpoint Hub says so in the UI).

## 8.2 Latent diffusion: why 448×640 becomes 56×80

Stable Diffusion doesn't denoise pixels. A **VAE** compresses a 448×640×3 image into a 56×80×4 latent (8× smaller
per side). Diffusion runs in that latent space, and the VAE decoder turns the result back into pixels.

**Consequences in this project:**
- **Small faces.** A face 40 px wide is only 5 latent pixels, too few for eyes and lips. The *face fix* (F10) crops the face, enlarges it to 512 px (64 latent pixels) and redraws it.
- **Latent scaling.** Latents are multiplied by 0.18215 (`vae.config.scaling_factor`) so they have roughly unit variance.
- **Swappable decoder.** The fine-tuned VAE `sd-vae-ft-mse` decodes the *same* latent space better. That's why it was swapped in without retraining (D-15).
- **Sizes must be multiples of 8** (a pydantic validator). In practice multiples of 64 are used, for the UNet's downsampling.

## 8.3 The UNet and cross-attention

The noise predictor is a UNet: it downsamples, processes and upsamples, with skip connections. Text enters through
**cross-attention** layers. In each one:

  Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

- Q (query) comes from the image features.
- K, V (keys, values) come from the 77 text-token embeddings.

So each image location "looks up" the words relevant to it. The projections `to_q`, `to_k`, `to_v` and `to_out` are
exactly where this project's LoRA is attached (D-03). They're the parts that connect words such as `<rvvarma>` to
visual features.

## 8.4 CLIP: text encoder and judge

CLIP was trained to put matching images and texts close together in a shared embedding space. It plays two roles here:

1. **Inside SD1.5:** its text encoder (ViT-L/14) turns the prompt into 77 token embeddings. **Hard limit:** 77 tokens including start and end. Anything after is silently dropped. That's why captions are capped at 69 + 6 (trigger) and prompts at 75, and why the 100-token negative prompt bug mattered (D-11; DECISIONS §4.1).
2. **As a judge (evaluation):** a separate, smaller CLIP ViT-B/32 embeds generated images, prompts and reference paintings. **Cosine similarity** measures closeness: cos(a, b) = a·b / (‖a‖‖b‖).

## 8.5 Classifier-free guidance

At each step the UNet runs twice: with the prompt and with the negative prompt (or empty). The final prediction
pushes away from the negative:

  ε = ε_neg + s · (ε_prompt − ε_neg),  with s = guidance_scale = 7.5

Higher s gives more literal prompt following but harsher images. This is also why a **negative prompt** works: it
replaces the "unconditional" branch, so the result moves away from "moustache, frame, blurry". It too is limited to
77 tokens.

## 8.6 LoRA: low-rank adaptation

A frozen weight matrix W (for example 320×320 in `to_q`) gets a trainable update:

  W' = W + (α / r) · B · A,  with A: r×320, B: 320×r, r = 16

- B starts at zero, so training starts from the unchanged base model (PEFT's Gaussian init sets A; B = 0).
- **Parameters:** 2·320·16 instead of 320² for this layer. Across the attention layers that's **3.2 M trainable** instead of 860 M (0.37 %).
- **Scale:** α = r = 16 gives a scale of 1. At generation time `lora_scale` multiplies the update again (`cross_attention_kwargs={"scale": …}`), which is the app's "style strength" slider.
- **Why it doesn't overfit as easily:** a rank-16 update can only move the weights in 16 directions per layer. That's enough for a style, and not enough to store 182 paintings. **Measured:** max similarity to any training image 0.898 < 0.95.

## 8.7 Training tricks used in v3

**Min-SNR-γ weighting.** The signal-to-noise ratio at step t is SNR(t) = ᾱₜ / (1 − ᾱₜ). Low-noise steps (high SNR) are
easy and dominate the gradient. The weighting:

  w(t) = min(SNR(t), γ) / SNR(t),  γ = 5

This caps their influence, which is reported in the literature to speed convergence (`_min_snr_weights`).

**Noise offset (0.05).** Adds a small per-channel constant to the noise, so the model can learn to shift an image's
overall brightness. Plain noise is zero-mean per image, which biases SD toward medium brightness. Varma's dark
backgrounds benefit.

**Caption dropout (0.1).** In 10 % of samples the caption is replaced by just `<rvvarma>`, so the trigger alone must
carry the style (`CaptionCollator._text`).

**Mixed precision.** The frozen model runs in fp16 to halve memory. The LoRA weights stay fp32, because small updates
would underflow in fp16 (`cast_training_params`).

**Gradient accumulation (4).** Four micro-batches are summed before one optimiser step, which simulates a larger
batch on a 16 GB GPU.

## 8.8 img2img and "strength"

Instead of starting from pure noise, img2img encodes an existing image, adds noise up to step `strength × T`, and
denoises from there:
- **Strength 0.4** (face and hires fix) keeps structure and refines detail.
- **Strength 0.55** (photo mode) repaints texture.
- **Strength 0.7+** without a lock invents a new picture. That's exactly the Spider-Man failure (D-20).

## 8.9 ControlNet

ControlNet is a trainable copy of the UNet's encoder that reads a **control image** (a pose skeleton or an edge map)
and adds its features into the frozen UNet through zero-initialised convolutions.
- The pre-trained `lllyasviel` ControlNets were trained on SD1.5, so they plug into *our* UNet + LoRA unchanged.
- **Division of labour:** the LoRA decides *how it looks*; ControlNet decides *where things are*.
- **Canny** keeps every outline. **OpenPose** keeps only 18 body joints per person.
- **Strength** (0.8 default) scales the ControlNet's contribution.

## 8.10 The metrics

| Metric | Definition | Reads as |
|---|---|---|
| Prompt adherence | cos(CLIP(image), CLIP(prompt without the trigger)) | Does it show what was asked? ~0.30–0.35 is typical for ViT-B/32 |
| Style similarity (mean / top-3) | Mean cosine to the 12 held-out paintings (or the 3 nearest) | Closer to real Varma paintings |
| Nearest-training similarity | Max cosine to any training image | > 0.95 would suggest copying |
| Diversity | Mean (1 − cos) between seeds of one prompt | Low = the same picture each time |
| PCK@0.1 | Share of reference joints whose generated counterpart lies within 0.1 × image size | Pose adherence (0 to 1) |

**Why paired:** CLIP scores mix content and style. Comparing LoRA and base on the *same* prompt and seed cancels
the content part, so the delta reflects the LoRA.

## 8.11 Perceptual hashing (de-duplication)

dHash shrinks an image to a tiny greyscale grid and records, for each pixel, whether it is brighter than its right
neighbour. That gives a bit string (256 bits here). Two copies of the same painting (a different scan, resize or
JPEG quality) differ in only a few bits.
- **Hamming distance** counts differing bits; ≤ 40 / 256 is treated as "the same painting".
- **Crops and recolourings** defeat it, which is why curated groups exist (D-07).

## 8.12 Haar cascades

OpenCV's Haar detectors slide a window of simple light/dark rectangle features over the image, with a cascade of
cheap classifiers rejecting non-faces early.
- **Fast, no extra model**, but tuned for frontal faces in photographs.
- **Hence the misses** on profiles, and false positives on round ornaments.
- **The eye cascade inside each face box** is a second check (`detect_faces`).
