# The Next Ravi Varma — User Manual

A complete guide for the team: **what** the project does, **how every piece works from first
principles**, **how to run it**, and **how to read the results**. Read Part A if you are new to
diffusion models; jump to Part B if you only need the project specifics.

Companion document for presentations: [VIVA_GUIDE.md](VIVA_GUIDE.md) (likely examiner questions + answers).

---

## Contents

- **Part A — Foundations (from basics)**
  1. [Images are tensors](#a1-images-are-tensors)
  2. [Neural networks and how they learn](#a2-neural-networks-and-how-they-learn)
  3. [Convolutions and the U-Net](#a3-convolutions-and-the-u-net)
  4. [Attention and transformers](#a4-attention-and-transformers)
  5. [CLIP: connecting text and images](#a5-clip-connecting-text-and-images)
  6. [Autoencoders and the VAE](#a6-autoencoders-and-the-vae)
  7. [Diffusion models: the core idea and the maths](#a7-diffusion-models-the-core-idea-and-the-maths)
  8. [Sampling: schedulers, steps and seeds](#a8-sampling-schedulers-steps-and-seeds)
  9. [Classifier-free guidance and negative prompts](#a9-classifier-free-guidance-and-negative-prompts)
  10. [Latent diffusion and Stable Diffusion 1.5](#a10-latent-diffusion-and-stable-diffusion-15)
  11. [Fine-tuning and LoRA](#a11-fine-tuning-and-lora)
  12. [Training mechanics: AdamW, warm-up, mixed precision, accumulation, checkpointing](#a12-training-mechanics)
  13. [Vision-language models: BLIP and VQA](#a13-vision-language-models-blip-and-vqa)
- **Part B — Our project, stage by stage**
  1. [The big picture](#b1-the-big-picture)
  2. [Repository map](#b2-repository-map)
  3. [Stage 1 — Downloading the corpus](#b3-stage-1--downloading-the-corpus)
  4. [Stage 2 — Validation and curation](#b4-stage-2--validation-and-curation)
  5. [Stage 3 — Captioning](#b5-stage-3--captioning)
  6. [Stage 4 — Preparation](#b6-stage-4--preparation)
  7. [Stage 5 — LoRA training](#b7-stage-5--lora-training)
  8. [Stage 6 — Generation](#b8-stage-6--generation)
  9. [Stage 7 — Evaluation](#b9-stage-7--evaluation)
- **Part C — Running it** ([local](#c1-local-mac--cpu) · [Colab](#c2-gpu-training-on-google-colab) · [tests](#c3-tests))
- **Part D — Results and how to read them**
- **Part E — Troubleshooting**
- **Part F — Glossary**

---

# Part A — Foundations (from basics)

## A1. Images are tensors

A colour image of width W and height H is stored as a grid of pixels, each with three numbers
(red, green, blue) between 0 and 255. To a neural network it is a **tensor** of shape
3 × H × W. Before training we rescale the values to [−1, 1]:

```
x = pixel / 127.5 − 1
```

This is what `transforms.Normalize([0.5],[0.5])` does in `src/ravi_varma/data/dataset.py`.
A batch of B images is a 4-D tensor B × 3 × H × W.

## A2. Neural networks and how they learn

A neural network is a function f(x; θ) with millions of adjustable numbers θ
(**parameters** or **weights**). The simplest building block is a **linear layer**:

```
y = W·x + b
```

where W is a d × k matrix. Stacking linear layers with non-linear **activations**
(ReLU, SiLU, GELU) between them lets the network represent complicated functions.

**Learning** means choosing θ to minimise a **loss** L(θ) — a number that
measures how wrong the network is on training examples. We compute the **gradient**
∇θ L (the direction in which the loss increases fastest) with
**back-propagation** and take a small step the other way:

```
θ ← θ − η · ∇θ L
```

η (eta) is the **learning rate**. One update is an **optimizer step**; one pass over the whole
dataset is an **epoch**. Because the full dataset rarely fits in memory, each step uses a small
random **batch** (stochastic gradient descent).

Key failure modes:
- **Underfitting**: the model cannot represent the pattern (too small, too little training).
- **Overfitting / memorisation**: the model reproduces training examples instead of the general
  pattern. With only 122 paintings this is the main risk — which is why we use LoRA (A11), hold out
  12 paintings and check for near-copies (B9).

## A3. Convolutions and the U-Net

A **convolution** slides a small filter (e.g. 3×3) over the image and computes a weighted
sum at each position. Early filters detect edges and colours; deeper ones detect textures, shapes,
faces. Convolutions are efficient because the same small filter is reused everywhere.

A **U-Net** is a convolutional network shaped like a "U":
- the **down path** repeatedly halves the resolution while increasing the number of channels
  (compressing *what* is in the image),
- the **up path** restores the resolution,
- **skip connections** copy features from the down path to the matching level of the up path so
  fine detail is not lost.

The U-Net's output has the **same shape as its input**. That is exactly what diffusion needs: the
input is a noisy image, the output is the noise it predicts (A7). Stable Diffusion's U-Net has
**~860 million parameters** and also contains attention layers (A4) and receives the timestep t
as an extra input.

## A4. Attention and transformers

**Attention** lets every element of a sequence look at every other element and decide what is
relevant. Each element produces three vectors — **query** Q, **key** K and **value** V —
through learned linear layers. The output is

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d ) · V
```

In words: compare each query with every key (dot product), turn the scores into weights with
softmax, and take the weighted average of the values. √d keeps the scores in a stable range.

- **Self-attention**: Q, K, V all come from the same sequence (image features attending to
  other image features — useful for global composition).
- **Cross-attention**: Q comes from the image features, K and V from the **text** embedding.
  This is how the prompt steers the image: each image location "asks" which words matter to it.

In code these four linear layers are called `to_q`, `to_k`, `to_v` and `to_out.0` (the output
projection). **These are exactly the layers our LoRA modifies** (A11, B7).

A **transformer** is a stack of attention + feed-forward blocks. The text encoder of Stable
Diffusion is a transformer.

## A5. CLIP: connecting text and images

**CLIP** (Contrastive Language–Image Pre-training, OpenAI 2021) has two encoders — one for images,
one for text — trained on ~400 million image–caption pairs so that a matching image and caption
produce **similar vectors** and non-matching ones produce dissimilar vectors. Similarity is the
**cosine similarity** of the normalised vectors:

```
cos(u, v) = (u · v) / (‖u‖ · ‖v‖)
```

We use CLIP in two different ways:
1. **Inside Stable Diffusion**: the CLIP **text encoder** (ViT-L/14 variant) turns the prompt into
   a sequence of 77 vectors of size 768 that feed the U-Net's cross-attention.
2. **For evaluation**: a separate, smaller CLIP model (ViT-B/32) measures how well an image matches
   its prompt and how similar it is to real Ravi Varma paintings (B9).

**Tokenisation.** Text is first split into **tokens** (sub-word pieces) using byte-pair encoding.
CLIP accepts at most **77 tokens** including a start and an end token; anything longer is silently
cut off. Our trigger word is split like this:

```
"<rvvarma>, a woman"  ->  ['<', 'rv', 'var', 'ma', '>', ',', 'a', 'woman']
```

So `<rvvarma>` is not one special token; it is a **rare sequence of five ordinary tokens** that never
appears in normal text — which is exactly why it works as a trigger (A11).

## A6. Autoencoders and the VAE

An **autoencoder** has an **encoder** that compresses an input into a small **latent** code and a
**decoder** that reconstructs the input from it. A **variational autoencoder (VAE)** makes the latent
space smooth and well-behaved by encoding to a probability distribution and sampling from it.

Stable Diffusion's VAE compresses an image by a factor of **8 in each direction** into **4 channels**:

| Image | Latent |
|---|---|
| 512 × 512 × 3 (786,432 numbers) | 64 × 64 × 4 (16,384 numbers) — 48× smaller |
| 448 × 640 × 3 (our most common size) | 56 × 80 × 4 |

The latents are multiplied by a fixed **scaling factor 0.18215** so they have roughly unit variance
(`latents * vae.config.scaling_factor` in our trainer).

## A7. Diffusion models: the core idea and the maths

**Idea:** destroying an image with noise is easy; learning to *undo* that destruction step by step
gives a generator. Start from pure noise, remove a little noise many times, and an image appears.

### Forward process (adding noise — fixed, no learning)

Take a clean image (or latent) x₀. Over T = 1000 timesteps we add Gaussian noise according to a
**noise schedule** β₁ … β_T (for SD1.5 a "scaled linear" schedule from 0.00085 to 0.012).
Define αₜ = 1 − βₜ and ᾱₜ = α₁ · α₂ · … · αₜ (the product of all αs up to t). A useful property lets us
jump straight to any timestep:

```
xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε        where ε ~ N(0, I)  (random Gaussian noise)
```

At small t, xₜ is almost the clean image; at t = 1000 it is almost pure noise.
This is `noise_scheduler.add_noise(latents, noise, timesteps)` in our trainer.

### Reverse process (removing noise — this is what we learn)

A network εθ(xₜ, t, c) — the U-Net — is trained to **predict the noise**
ε that was added, given the noisy input xₜ, the timestep t and the text condition c.

### Training objective

```
L = E[ ‖ ε − εθ(xₜ, t, c) ‖² ]      (average over images x₀, noise ε and timesteps t)
```

i.e. the **mean squared error (MSE)** between the true and predicted noise. One training step:

1. take a training image + caption,
2. encode the image to a latent x₀ with the VAE, encode the caption with CLIP to c,
3. pick a random timestep t between 0 and 999 and random noise ε,
4. build xₜ with the formula above,
5. predict ε̂ = εθ(xₜ, t, c),
6. loss = MSE(ε̂, ε), back-propagate, update.

**Why our loss curve is flat (~0.17–0.18):** because t is random every step, some steps are
almost-clean images (very hard to guess the tiny noise) and some are almost-pure noise (easy). The
loss mostly reflects *which* t was drawn, not how well the style is learned. The **validation
images** are the real progress signal (see Part D).

## A8. Sampling: schedulers, steps and seeds

To generate, we start from random noise x_T and repeatedly use the predicted noise to move towards
a clean latent. The original DDPM sampler needs ~1000 steps. Modern **schedulers / samplers** treat
the reverse process as solving a differential equation and take larger, smarter steps:

- **DPM-Solver++ (multistep)** — our default; good images in **20–30 steps**.
- Euler-ancestral, DDIM, PNDM are alternatives (`model.scheduler` in `configs/generation.yaml`).

The final latent is decoded by the VAE into pixels.

**Seed.** The starting noise comes from a random number generator. The same seed + prompt + settings
gives the same image. We create the noise with a **CPU generator** so a seed gives the same starting
noise on CUDA, Apple MPS or CPU — this is what makes our "with vs without LoRA" comparisons fair.

## A9. Classifier-free guidance and negative prompts

At every sampling step the U-Net is run **twice**: once with the prompt (c) and once with an
"unconditional" input (∅, normally the empty prompt). The two predictions are combined:

```
ε̂ = εθ(xₜ, ∅) + s · ( εθ(xₜ, c) − εθ(xₜ, ∅) )
```

s is the **guidance scale** (we use 7.5). s = 1 means no extra guidance; larger s follows the
prompt more strongly but can over-saturate.

A **negative prompt** simply replaces the empty unconditional input with a description of what we
do *not* want ("deformed hands, extra fingers, blurry, frame, watermark…"). The formula then pushes
the image **away** from it.

## A10. Latent diffusion and Stable Diffusion 1.5

Running diffusion directly on 512×512×3 pixels is expensive. **Latent diffusion** (Rombach et al.,
2022) runs it in the VAE's small latent space instead. **Stable Diffusion 1.5** therefore has three
parts:

| Component | What it does | Size | Trained by us? |
|---|---|---|---|
| CLIP text encoder | prompt → 77 × 768 embedding | ~123 M params | No (frozen) |
| U-Net | predicts noise in latent space, conditioned on text via cross-attention | ~860 M params | **Only LoRA adapters** |
| VAE | image ↔ latent (8× down/up-sampling) | ~84 M params | No (frozen) |

Generation pipeline: prompt → text encoder → (U-Net denoising loop from random latent, guided by
cross-attention + CFG) → VAE decoder → image.

SD1.5 was trained mainly at **512×512**, which is why our training buckets keep roughly 512² pixels.
We use the official mirror `stable-diffusion-v1-5/stable-diffusion-v1-5` (the original
`runwayml/...` repository was removed from Hugging Face in 2024; the weights are identical).

## A11. Fine-tuning and LoRA

**Fine-tuning** continues training a pre-trained model on new data. Options for teaching a style:

| Method | What is trained | Size of result | Notes |
|---|---|---|---|
| Full fine-tune / DreamBooth | all ~860 M U-Net weights | 2–4 GB | needs lots of GPU memory; easy to overfit/forget |
| Textual inversion | one new word embedding | a few KB | weak for a whole painting style |
| **LoRA (ours)** | small low-rank matrices added to chosen layers | **12.8 MB** | cheap, keeps base knowledge, can be scaled or switched off |

### The LoRA maths

Take a frozen weight matrix W₀ of size d × k (e.g. the `to_q` projection). LoRA
(Hu et al., 2021) learns an **update of low rank r**:

```
W = W₀ + ΔW,     ΔW = (α / r) · B · A,     B is d × r,  A is r × k
```

so the layer computes `h = W₀·x + (α/r) · B·(A·x)`.

- **Why it is small:** a 320 × 320 projection has 102,400 weights; a rank-16 LoRA for it
  has 16 × (320 + 320) = 10,240 — **10×** fewer. Over the whole U-Net our adapter has
  **3,188,736** trainable parameters = **0.37 %** of the U-Net.
- **Why it is safe:** B starts at **zero**, so at step 0 the model is exactly the base model; it
  only drifts as far as the data pushes it. W₀ never changes.
- **Rank r** = capacity of the update (we use 16). **Alpha α** sets the scale
  α/r; with α = r = 16 the scale is 1.
- **Target modules:** `to_q, to_k, to_v, to_out.0` in every self- and cross-attention block. Cross-
  attention is where words meet image features, so this is where "`<rvvarma>` means *this* look" is
  learned; self-attention captures composition and texture relationships.
- **LoRA dropout 0.05** randomly zeroes 5 % of the adapter's inputs during training (regularisation).
- **LoRA scale at inference** (`--lora-scale`) multiplies ΔW: 0 = base model, 1 = as trained.

### The trigger token

Every training caption starts with `<rvvarma>`. The model learns to associate that rare token
sequence with whatever is **common to all images but not described in the captions** — the
painterly technique, palette, lighting, staging. That is why our captions deliberately describe
*content* (people, clothes, setting) and **not** style words like "oil painting": anything we
describe in words gets attached to those words instead of to the trigger. (This is also why we
*name* frames in captions — so "frame" absorbs the frame, not the trigger.) We keep the text encoder
frozen; the adapter learns how the U-Net should respond to those tokens.

## A12. Training mechanics

- **AdamW optimizer.** Adam keeps running averages of each parameter's gradient (momentum,
  β₁ = 0.9) and squared gradient (β₂ = 0.999) and scales each step individually;
  "W" adds decoupled **weight decay** (0.01) that gently pulls weights toward zero. ε = 10⁻⁸
  avoids division by zero.
- **Learning rate 1e-4, constant with 100 warm-up steps.** The LR ramps linearly from 0 to 1e-4 over
  the first 100 optimizer steps (avoids large unstable updates while Adam's statistics are still
  empty), then stays constant. 1e-4 is a standard LoRA LR; full fine-tuning uses ~1e-5 or less.
- **Batch size 1 × gradient accumulation 4 = effective batch 4.** We compute gradients for 4 images
  one at a time, add them up, and update once. Same maths as batch 4, a quarter of the memory.
  One **optimizer step** = 4 images, so 1500 steps = 6000 images ≈ **49 epochs** of 122 images.
- **Gradient clipping 1.0** caps the size of the gradient to prevent rare huge updates.
- **Mixed precision (fp16).** The frozen weights are stored in 16-bit floats (half the memory, faster
  on the T4's tensor cores); the **LoRA weights stay in fp32** so tiny updates are not rounded away.
  A **gradient scaler** multiplies the loss before back-prop so small fp16 gradients do not underflow
  to zero, then un-scales before the update.
- **Gradient checkpointing.** Instead of storing every intermediate activation for back-prop, the
  U-Net recomputes some of them — ~30 % slower, much less memory.
- **Checkpoints** every 250 steps save the full training state (to resume after a Colab disconnect)
  **and** a loadable LoRA file (to compare checkpoints).
- **Seed 42** fixes all randomness (data order, noise, timesteps) for reproducibility.

## A13. Vision-language models: BLIP and VQA

**BLIP** (Salesforce) combines a Vision Transformer image encoder with a text decoder.
- `blip-image-captioning-large` writes a free-form sentence for an image (we use **beam search** with
  3 beams — it keeps the 3 most likely partial sentences at each word and returns the best).
- `blip-vqa-base` does **Visual Question Answering**: given an image and a question, it generates a
  short answer ("what is the main person wearing?" → "sari").

VLMs make mistakes ("tie", "phone" in a 19th-century painting), so we gate, filter and store all
answers (B5).

---

# Part B — Our project, stage by stage

## B1. The big picture

```
Wikimedia Commons ─► download ─► validate & curate ─► caption (BLIP + VQA) ─► prepare (buckets)
                                                                                  │
                    ┌─────────────────────────────── manifest.jsonl (122) ◄───────┘
                    ▼                                  reference/ (12 held out)
         LoRA training on SD1.5 (Colab T4) ─► pytorch_lora_weights.safetensors
                    │
                    ▼
     generate (prompt + <rvvarma>) ─► images ─► evaluate vs base model + held-out paintings
```

- **Milestone 1** (done): everything up to the manifest.
- **Milestone 2** (done): training, generation, evaluation.
- **Milestone 3** (next): an LLM turns a story ("Yudhishthira weeping over Karna") into a detailed prompt.
- **Milestone 4** (next): ControlNet pose control, fuller evaluation, Gradio web UI.

## B2. Repository map

| Path | What it contains |
|---|---|
| `configs/dataset.yaml` | folders, validation thresholds, **curation decisions** (exclusions, duplicate groups, hold-out list, caption notes), preprocessing |
| `configs/lora_sd15.yaml` | every training hyper-parameter |
| `configs/generation.yaml` | inference defaults, negative prompt, hardware settings |
| `configs/prompts/sample_prompts.yaml` | the 5 fixed evaluation prompts |
| `src/ravi_varma/config.py` | typed loading of the YAML files (pydantic) |
| `src/ravi_varma/data/metadata.py` | read/write `metadata.jsonl`, title cleaning |
| `src/ravi_varma/data/validation.py` | the validator: accepted/excluded decisions, manifest check |
| `src/ravi_varma/data/captions.py` | BLIP + VQA captioner, filters, caption assembly |
| `src/ravi_varma/data/dataset.py` | preparation (buckets), PyTorch dataset, bucket sampler |
| `src/ravi_varma/utils/image.py` | safe image loading, hashing, border trimming, buckets |
| `src/ravi_varma/utils/device.py` | CUDA / MPS / CPU detection, memory settings |
| `src/ravi_varma/training/lora_trainer.py` | the training loop |
| `src/ravi_varma/training/callbacks.py` | checkpoints, validation images, loss logging |
| `src/ravi_varma/generation/pipeline.py` | loads SD1.5 + LoRA and generates |
| `src/ravi_varma/evaluation/` | CLIP metrics and the run-level evaluator |
| `scripts/*.py` | command-line entry points for every stage |
| `scripts/colab_train.sh` | one-shot GPU run (install → train → generate → evaluate → zip) |
| `notebooks/03_lora_training.ipynb` | the Colab notebook version of the GPU run |
| `data/metadata/`, `data/captions/`, `data/processed/manifest.jsonl`, `data/reference/reference.jsonl` | the dataset's text files (in git) |
| `outputs/generated/`, `outputs/evaluations/` | images and evaluation reports (in git) |
| `checkpoints/lora/ravi_varma/` | trained weights (**not** in git — keep in Drive) |
| `tests/` | 101 automated tests |
| `generation/prompt_engine.py`, `generation/controlnet.py`, `app/` | early prototypes for Milestones 3–4 (not part of the completed work) |

## B3. Stage 1 — Downloading the corpus

`scripts/download_dataset.py` uses the **Wikimedia Commons API** to walk the category
*"Paintings by Raja Ravi Varma"* and its sub-categories, downloads the files and records for each one:
the exact Commons file name, licence, cleaned title, explicit year (only if stated in the source) and
the **original pixel size**. Wikimedia rate-limits downloads, so the script waits and retries.

`--from-metadata` re-downloads **exactly** the files listed in the committed `metadata.jsonl`, so anyone
can rebuild the identical raw dataset (the images themselves are not stored in git).

Result: **200 files**, 193 public domain; the others CC BY / CC BY-SA / GODL-India (licence per image in metadata).

## B4. Stage 2 — Validation and curation

`scripts/validate_dataset.py` → `DatasetValidator.validate()` makes an **accepted / excluded + reason**
decision for every file and saves it to `data/metadata/validation_report.json`.

Checks, in order:
1. **Format and readability** — unsupported extensions, corrupt or truncated files.
2. **Resolution** — shortest side ≥ 384 px, checked against the **original Commons size**. We found
   26 local files that had been **upscaled** to exactly 384 px from originals as small as 200 px;
   upscaling adds blur, not detail, so they are excluded.
3. **Curated exclusions** (`configs/dataset.yaml`) — 10 files that are not painting reproductions:
   postage stamps, museum-gallery photos, multi-panel grids, prints with large text labels.
4. **Duplicates** — the same artwork often exists as an original scan, an oleograph print and a crop:
   - **exact duplicates**: identical file hash (SHA-256);
   - **curated duplicate groups**: 22 groups found by looking at contact sheets (CLIP similarity was
     used only as a review aid);
   - **automatic near-duplicates**: **difference hash (dHash)** — shrink to 17×16 grey pixels, compare
     each pixel to its right neighbour → 256 bits; two images whose bit strings differ in ≤ 40
     positions (**Hamming distance**) are treated as the same artwork.
   In each group the **highest-resolution** copy is kept.
5. **Captions present** (checked again before preparation).

Result: **134 accepted**, 66 excluded (26 upscaled, 7 non-paintings*, 33 duplicates).
\*10 are on the curated list; 3 of them were already excluded as upscaled.

## B5. Stage 3 — Captioning

`scripts/generate_captions.py` captions every accepted image (≈ 13 s/image on a MacBook):

1. **BLIP-large** writes a sentence ("a woman in a red sari feeding a white swan").
2. **BLIP-VQA** answers **12 fixed questions**: people count, clothing, clothing colour, wearing
   jewelry?, which jewelry, posture, action, expression, holding something?, what object, setting,
   day or night. Follow-ups are **gated** — "what jewelry?" is asked only if "wearing jewelry?" = yes.
3. **Measured properties** from pixels: orientation, overall brightness, dominant colour family.
4. **Filters**: answers that describe the medium ("in painting", "museum") or are impossible in a
   19th-century painting ("phone", "smoking", "laptop") are dropped; "guitar" → "stringed instrument".
5. **Assembly** in priority order — title (from provenance), sentence, clothing, jewelry, pose,
   expression, objects, setting, composition, lighting, palette — dropping the lowest-priority parts
   until the caption fits in **70 tokens** (leaving room for `<rvvarma>, ` and the start/end tokens
   inside CLIP's 77).

Every question and answer is saved in `data/captions/<image>.json`. If you fix an answer by hand or
change a rule, `python scripts/generate_captions.py --reflatten` rebuilds all captions from the saved
answers without re-running the models.

Example (real):

> `<rvvarma>, Hamsa Damayanti, a woman in a red sari feeding a white swan, wearing red dress, necklace jewelry, sitting, calm expression, in park, single-figure composition, vertical format, mid-tone lighting, warm red and ochre palette`

Note the remaining imprecision ("red dress" for a sari, "park" for a palace garden) — typical of VQA.

## B6. Stage 4 — Preparation

`scripts/prepare_dataset.py` re-runs validation and then, for each accepted image:
1. **Trims flat borders** (white scanner margins) — rows/columns with almost no variation that match
   the corner colour, at most 15 % per side.
2. **Aspect-ratio buckets** instead of square crops. Most Varma canvases are tall (≈ 1:1.4); a square
   512 crop cuts off heads and feet. We pre-computed 11 sizes with ≈ 512² pixels (±15 %), sides that
   are multiples of 64, ratio ≤ 1.75 — e.g. **448×640** (81 images), 448×576, 512×512, 640×448 — and
   resize each image to the closest one with a minimal crop.
3. Writes `data/processed/<name>.jpg` and a line in `manifest.jsonl` with the caption
   (`<rvvarma>, ...`), size and provenance.
4. The **12 hold-out paintings** go to `data/reference/` instead — never seen in training, used only
   for evaluation.

During training a **bucket batch sampler** only puts images of the same size in a batch (with our
batch size of 1 this is simply a shuffle).

`python scripts/validate_dataset.py --manifest` re-checks exactly what training will read: every image
readable with sides divisible by 8, every caption starting with the trigger, ≤ 77 tokens, no placeholders.

## B7. Stage 5 — LoRA training

Script: `scripts/train_lora.py --config configs/lora_sd15.yaml` (runs on a CUDA GPU; we used a free
Colab **Tesla T4**). What happens:

1. Load SD1.5's tokenizer, text encoder, VAE, U-Net and noise scheduler.
2. **Freeze** everything; move frozen weights to **fp16**.
3. Attach **LoRA** (rank 16, alpha 16, dropout 0.05) to `to_q/to_k/to_v/to_out.0`; keep LoRA weights
   in fp32. The trainer **checks** that only parameters named `lora` are trainable.
4. Every step (×4 for accumulation): load image + caption → VAE latent → CLIP text embedding → random
   noise + timestep → noisy latent → U-Net predicts noise → MSE loss → back-prop through the LoRA weights only.
5. Every optimizer step: clip gradients, AdamW update, LR schedule step, log the mean loss.
6. Every 250 steps: save a checkpoint (+ loadable LoRA file) and render **3 fixed validation prompts
   with a fixed seed** → `outputs/logs/validation_images/step_XXXXX/`. Step 0 is rendered before
   training as a baseline.
7. At the end: save `pytorch_lora_weights.safetensors`, `training_config.yaml`,
   `training_summary.json`, `loss_history.csv`.

| Setting | Value |
|---|---|
| Base model | Stable Diffusion 1.5 (frozen) |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 |
| Target modules | to_q, to_k, to_v, to_out.0 |
| Optimizer | AdamW (β 0.9/0.999, weight decay 0.01) |
| Learning rate | 1e-4, constant, 100 warm-up steps |
| Batch / accumulation / effective batch | 1 / 4 / 4 |
| Steps | 1500 (≈ 49 epochs of 122 images) |
| Precision | fp16 mixed precision, gradient checkpointing |
| Seed | 42 |
| Time | 58 min on a T4 (≈ 2.2 s per optimizer step) |
| Output | 12.8 MB LoRA file + 6 checkpoints |

`--smoke-test` runs 20 steps into `outputs/smoke_test/` first — on Colab it caught a library
incompatibility (bitsandbytes) in one minute instead of after an hour. If Colab disconnects, re-running
the same command **resumes** from the latest checkpoint.

## B8. Stage 6 — Generation

`scripts/generate.py` loads SD1.5 + our LoRA and renders images:

```bash
python scripts/generate.py --prompt "<rvvarma>, royal Indian portrait, jewelry, classical oil painting" --seed 42
python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44 --output-dir outputs/generated/lora
python scripts/generate.py --prompt-file configs/prompts/sample_prompts.yaml --seeds 42 43 44 --no-lora --output-dir outputs/generated/baseline
```

Defaults: DPM-Solver++ 30 steps, guidance 7.5, LoRA scale 1.0, a negative prompt against bad anatomy,
blur, text, watermarks and frames. Useful options: `--lora-scale` (style strength), `--lora-path`
(a specific checkpoint), `--no-lora` (base model for comparison), `--width/--height`, `--steps`,
`--guidance-scale`. Every image gets a `.json` sidecar with all settings so it can be regenerated.

On Apple-silicon Macs we use fp16 **without attention slicing** — the combination of fp16 and sliced
attention produced NaN values (all-black images) with PyTorch 2.4.

## B9. Stage 7 — Evaluation

`scripts/evaluate.py --input outputs/generated/lora --baseline outputs/generated/baseline` computes,
for the 5 prompts × 3 seeds = **15 pairs** (same prompt and seed with and without LoRA):

| Metric | What it measures | Base | LoRA |
|---|---|---|---|
| Prompt adherence | CLIP cosine between image and prompt (trigger removed) | 0.328 | 0.317 |
| Style similarity (mean) | average CLIP image similarity to the **12 held-out** paintings | 0.669 | 0.681 |
| Style similarity (top-3) | similarity to the 3 closest held-out paintings | 0.734 | 0.746 |
| Diversity | average CLIP distance between the 3 seeds of a prompt | 0.106 | 0.119 |
| Nearest training image | memorisation check (> 0.95 would be a near-copy) | — | 0.884 max |

It also writes contact sheets and `qualitative_review.csv` — a sheet for **human 1–5 ratings** of
visual quality, anatomy, style and prompt adherence, because CLIP cannot judge hands, faces or
painting technique. Limitations are explained in [EVALUATION.md](EVALUATION.md).

---

# Part C — Running it

## C1. Local (Mac / CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/download_dataset.py --from-metadata   # ~1 h because of Wikimedia rate limits
python scripts/validate_dataset.py
python scripts/generate_captions.py                  # ~30 min on a MacBook
python scripts/prepare_dataset.py
python scripts/validate_dataset.py --manifest
python scripts/package_dataset.py                    # -> dist/ravi_varma_dataset.zip for Colab
```

With trained weights in `checkpoints/lora/ravi_varma/`, generation also works locally (~1 min/image on an
8 GB MacBook).

## C2. GPU training on Google Colab

1. Runtime → Change runtime type → **T4 GPU**, **Runtime version 2026.07** (the newest Colab image is
   Python 3.13, for which our pinned libraries have no packages).
2. Open `notebooks/03_lora_training.ipynb` and run top to bottom, *or* upload the code + dataset and run
   `bash scripts/colab_train.sh`.
3. Download `ravi_varma_results.zip` and `unzip -o ravi_varma_results.zip` in the project root.

## C3. Tests

`pytest -v` — 101 tests (99 run; 2 only run on machines without CLIP). They cover validation and
curation, captions, buckets, title cleaning, generation settings, evaluation and a **real training run
on a 1 MB test model** (checks freezing, checkpointing, resume and that the exported LoRA loads).

---

# Part D — Results and how to read them

- **Training progression** (`outputs/evaluations/lora_vs_base/validation_progression.jpg`): at step 0 the
  validation prompts give framed, flat, poster-like images; after ~250 steps the rendering becomes painted;
  after step 1000 the compositions become Varma-like (seated royal woman with red curtains, pillared court).
  The style was still improving at step 1500, so the final checkpoint is used.
- **Base vs LoRA** (`side_by_side.jpg`): with the LoRA — full-figure staging with pillars, carpets and lamp
  stands, maharaja-style portraits, darker painted backgrounds, and **no picture frames (0/15 vs 5/15)**.
- **Problems found**: a dark moustache-like shadow on some women's upper lips; signature-like marks in
  corners (many training prints carry signatures); crowded mythological scenes still look like posters.
- **Metrics**: only small CLIP changes despite clear visual differences — CLIP ViT-B/32 is a weak judge of
  painting style. No memorisation.

---

# Part E — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| All-black images on a Mac | fp16 + attention slicing → NaN on MPS | keep `attention_slicing: "auto"` (off on MPS); the generator refuses black images |
| Very slow generation on a Mac, heavy swap | 8 GB RAM; fp32 or other apps | use the default fp16; close other apps; one seed per run |
| `No module named 'triton.ops'` on Colab | old bitsandbytes with new Triton | do not install bitsandbytes (already removed) |
| pip downgrades torch on Colab | an xformers pin requiring torch 2.4 | do not install xformers (already removed) |
| Colab install fails on Python 3.13 | pinned packages have no 3.13 wheels | Runtime version **2026.07** |
| torchao import error | Colab torchao/torch mismatch | the notebook's step 3 pins or removes torchao |
| `lora_loaded: false` in outputs | weights not in `checkpoints/lora/ravi_varma/` | unzip the Colab results or pass `--lora-path` |
| `git push` 403 | saved token only covers the old repo | `gh auth login` then `gh auth setup-git` |
| Captions contain "unknown (no VLM…)" | BLIP failed to load | the captioner now stops instead; check internet / disk space |

---

# Part F — Glossary

| Term | Meaning |
|---|---|
| Accumulation | adding gradients from several mini-batches before one update |
| Alpha (LoRA) | scale of the LoRA update (α/r) |
| Attention | weighted mixing of information based on query–key similarity |
| Bucket | one of the fixed training image sizes with ≈ 512² pixels |
| CFG / guidance scale | how strongly the prompt steers each denoising step |
| Checkpoint | saved training state at a given step |
| CLIP | model that maps images and text into a shared vector space |
| Cross-attention | image features attending to text tokens |
| dHash | perceptual hash from neighbouring-pixel comparisons, for near-duplicate detection |
| Diffusion model | generator that learns to reverse gradual noising |
| Epoch | one pass over the training set |
| fp16 / mixed precision | 16-bit storage/compute for speed and memory, fp32 where precision matters |
| Hold-out / reference set | the 12 paintings never used for training |
| Latent | the VAE's compressed 4-channel representation (8× smaller per side) |
| LoRA | low-rank adapter: W₀ + (α/r)·B·A |
| Manifest | `manifest.jsonl`: list of training images + captions |
| Negative prompt | text the sampler is pushed away from |
| Optimizer step | one weight update (= 4 images here) |
| Rank (LoRA) | inner dimension r of the update; its capacity |
| Scheduler / sampler | algorithm that turns noise into an image in N steps |
| Seed | number fixing the random starting noise |
| Trigger token | `<rvvarma>`; the words that switch the learned style on |
| U-Net | the noise-predicting network at the heart of Stable Diffusion |
| VAE | the encoder/decoder between pixels and latents |
| VQA | visual question answering |
