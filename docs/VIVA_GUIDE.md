# Viva Guide — The Next Ravi Varma

Likely examiner questions with short answers you can say out loud. Each answer is 2–4 sentences;
the [User Manual](USER_MANUAL.md) has the full explanation behind each one (section in brackets).
Tip: answer in one sentence first, then add the detail only if they ask.

---

## Numbers to remember

| | |
|---|---|
| Images downloaded / accepted / trained / held out | **200 / 134 / 122 / 12** |
| Excluded | **66** = 26 upscaled low-res + 33 duplicates + 7 non-paintings |
| Base model | **Stable Diffusion 1.5** (U-Net ~860 M params, CLIP text encoder ~123 M, VAE ~84 M) |
| LoRA | rank **16**, alpha **16**, dropout **0.05**, on `to_q, to_k, to_v, to_out.0` |
| Trainable parameters | **3.19 M = 0.37 %** of the U-Net; LoRA file **12.8 MB** |
| Training | **1500** optimizer steps, batch 1 × accumulation 4, LR **1e-4** (100 warm-up), AdamW, fp16, seed 42 |
| Epochs | 1500 × 4 / 122 ≈ **49** |
| Hardware / time | Colab **Tesla T4**, **58 min** (~2.2 s/step) |
| Inference | DPM-Solver++ **30 steps**, guidance **7.5**, LoRA scale 1.0 |
| Latent size | image ÷ 8 per side, 4 channels (448×640 → 56×80×4) |
| CLIP token limit | **77** (captions capped at 70 + trigger) |
| Evaluation | 5 prompts × 3 seeds = **15 pairs**; style sim. 0.669 → **0.681**; prompt adherence 0.328 → 0.317; frames 5/15 → **0/15** |

---

## 1. Motivation and scope

**Q: What is the project in one line?**
We fine-tune Stable Diffusion 1.5 with a LoRA adapter on a curated set of Raja Ravi Varma paintings so
that a text prompt with the trigger word `<rvvarma>` generates new images in his style.

**Q: Why Ravi Varma?**
His style is distinctive and well documented: European academic oil technique (realistic anatomy,
chiaroscuro, soft modelling of skin and drapery) applied to Indian mythology and royal portraits. His works
are public domain, so a legal, provenance-tracked dataset is possible.

**Q: How is this related to "The Next Rembrandt"?**
Same goal — learn a painter's style from his body of work and create new pieces. The Next Rembrandt (2016)
used hand-engineered statistics of facial features and 3D-printed texture; we use a modern generative
diffusion model fine-tuned with parameter-efficient methods.

**Q: What is done and what is left?**
Milestones 1 (dataset + captions) and 2 (LoRA training, generation, evaluation) are complete — about 50 %.
Milestone 3 is LLM prompt expansion; Milestone 4 is ControlNet pose control, a fuller evaluation and a web UI.

---

## 2. Diffusion basics [A7–A10]

**Q: How does a diffusion model generate an image?**
During training we add Gaussian noise to images at random levels and train a network to predict the
added noise. To generate, we start from pure noise and repeatedly subtract the predicted noise over
~30 steps until a clean image remains.

**Q: Write the forward process.**
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$, with $\varepsilon \sim \mathcal N(0,I)$
and $\bar\alpha_t$ the cumulative product of $(1-\beta_t)$ over a fixed noise schedule of 1000 steps.

**Q: What is the training loss?**
Mean squared error between the true noise and the predicted noise:
$\mathbb E\,\lVert \varepsilon - \varepsilon_\theta(x_t, t, c)\rVert^2$, with a random timestep each step.

**Q: What are the three components of Stable Diffusion?**
A CLIP text encoder (prompt → 77×768 embedding), a U-Net that predicts noise in latent space and reads
the text through cross-attention, and a VAE that converts between pixels and the 8×-smaller latent space.

**Q: Why "latent" diffusion?**
Running diffusion on a 64×64×4 latent instead of 512×512×3 pixels is ~48× less data per step, so it is
far cheaper while the VAE decoder restores the detail.

**Q: What is a U-Net and why is it used?**
An encoder–decoder CNN with skip connections whose output has the same shape as its input — exactly
what is needed to predict a noise map for a noisy latent. SD's U-Net also has attention layers and takes
the timestep as input.

**Q: Explain attention / cross-attention.**
$\text{softmax}(QK^\top/\sqrt d)\,V$. In cross-attention the queries come from image features and the
keys/values from the text tokens, so each image region picks the words relevant to it — that is how
the prompt controls the image.

**Q: What is classifier-free guidance?**
The U-Net is run with and without the prompt and the prediction is extrapolated:
$\varepsilon_u + s(\varepsilon_c - \varepsilon_u)$. We use $s = 7.5$. A negative prompt replaces the
"without prompt" branch, so the image is pushed away from it.

**Q: What does the scheduler do? Why 30 steps and not 1000?**
It decides how to move from noise to image. DPM-Solver++ is a higher-order ODE solver that takes large,
accurate steps, so 20–30 steps give quality comparable to ~1000 DDPM steps.

**Q: What does the seed do?**
It fixes the random starting noise. Same seed + prompt + settings = same image; we use a CPU generator
so seeds are reproducible across GPU, Mac and CPU, which makes our base-vs-LoRA comparison fair.

---

## 3. LoRA and fine-tuning [A11]

**Q: What is LoRA?**
Low-Rank Adaptation: instead of changing a weight matrix $W_0$, we learn a low-rank update
$\Delta W = \frac{\alpha}{r} BA$ with $B\in\mathbb R^{d\times r}$, $A\in\mathbb R^{r\times k}$, $r \ll d,k$.
$W_0$ stays frozen; only $A$ and $B$ are trained.

**Q: Why LoRA instead of full fine-tuning or DreamBooth?**
Full fine-tuning updates ~860 M parameters, needs much more GPU memory, gives a 2–4 GB file and easily
overfits or forgets with only 122 images. LoRA trains 3.2 M parameters (0.37 %), fits a free T4, gives a
12.8 MB file, preserves the base model's knowledge and can be scaled or turned off at inference.

**Q: Why not textual inversion?**
Textual inversion only learns one new word embedding; it cannot change how the U-Net renders, so it is
weak for a whole painting style. LoRA changes the attention layers themselves.

**Q: Why rank 16? What does alpha do?**
Rank is the adapter's capacity: 4–8 can under-fit a full style, 64+ adds parameters and overfitting risk
on a small dataset; 16 is a common middle ground for style LoRAs. Alpha scales the update by $\alpha/r$;
with alpha = rank the scale is 1, so the learning rate behaves predictably.

**Q: Why these target modules?**
`to_q, to_k, to_v, to_out.0` are the attention projections. Cross-attention is where text meets image,
so the trigger word's meaning is learned there; self-attention shapes composition and texture.

**Q: How many parameters does LoRA save for one layer?**
For a 320×320 projection: full = 102,400; rank-16 LoRA = 16 × (320 + 320) = 10,240 — 10× fewer.

**Q: Why does LoRA not destroy the model at the start of training?**
$B$ is initialised to zero, so $\Delta W = 0$ and the model starts exactly as the base model.

**Q: What is the trigger token and how does it work?**
`<rvvarma>` starts every training caption. CLIP splits it into the rare tokens `<, rv, var, ma, >`, which
never occur in normal text, so the adapter learns to associate them with whatever is common to all
paintings but not described in the captions — the style.

**Q: Why are style words left out of the captions?**
Anything described in words gets attached to those words. If every caption said "oil painting", the
style would be learned by "oil painting" instead of the trigger. We describe only content, and we even
*name* frames where present so frames are not absorbed by the trigger.

**Q: Did you train the text encoder?**
No, it is frozen (`train_text_encoder: false`). Only the U-Net's attention layers get adapters — less
memory, less overfitting.

---

## 4. Training details [A12, B7]

**Q: Walk me through one training step.**
Image → VAE latent; caption → CLIP embedding; pick random timestep $t$ and noise; make the noisy latent;
the U-Net (with LoRA) predicts the noise; MSE loss; back-propagate into the LoRA weights only; after 4
such images, clip gradients and take one AdamW step.

**Q: Why learning rate 1e-4? Why warm-up?**
1e-4 is the standard range for LoRA (full fine-tuning uses ~1e-5 or lower because it changes all
weights). Warm-up ramps the LR from 0 over 100 steps so the first updates, made while Adam's statistics
are still empty, are not too large.

**Q: Why batch size 1 with gradient accumulation 4?**
Same effective batch of 4 as a real batch of 4, but only one image in memory at a time — necessary on a
16 GB T4 at ~512² resolution.

**Q: What is mixed precision? Any risk?**
Frozen weights are stored in fp16 (half memory, faster tensor cores); LoRA weights stay fp32 so small
updates are not rounded away. A gradient scaler multiplies the loss so tiny fp16 gradients do not
underflow to zero.

**Q: What is gradient checkpointing?**
Recomputing some activations during back-prop instead of storing them — about 30 % slower but much less memory.

**Q: Why 1500 steps? Did it overfit?**
1500 × 4 = 6000 images ≈ 49 epochs, a typical range for a ~100-image style LoRA. Validation images were
still improving at step 1500 and no generated image is a near-copy of a training image (max CLIP similarity
0.88, threshold 0.95), so we see no overfitting. Checkpoints every 250 steps let us pick an earlier one if needed.

**Q: Your loss curve is flat — did training fail?**
No. The loss is dominated by the random timestep: very noisy inputs are easy, nearly clean ones are hard,
so the average stays around 0.17–0.18. Diffusion LoRA loss is a poor progress signal; that is why we render
fixed validation prompts every 250 steps, and those show clear progress.

**Q: How did you make training reproducible and robust?**
Fixed seed 42, all settings in YAML, checkpoints every 250 steps with automatic resume, a 20-step smoke test
before the full run, and the exact training config saved next to the weights.

---

## 5. Dataset and captions [B3–B6]

**Q: Where does the data come from? Is it legal?**
Wikimedia Commons, category "Paintings by Raja Ravi Varma". He died in 1906, so the works are public domain;
193 of the 200 files are public-domain reproductions and the rest are CC BY / CC BY-SA / GODL-India. Each
file's source and licence is recorded, and images are not redistributed in our repository.

**Q: Why only 122 training images? Isn't that too few?**
LoRA needs far fewer images than full training — style LoRAs are commonly trained on 20–200. Quality matters
more than quantity: we removed 66 problem files. Adding blurry upscales and duplicates would teach blur and
over-weight a few paintings.

**Q: How did you find duplicates?**
Exact duplicates by SHA-256 file hash; near-duplicates by a 256-bit difference hash (dHash) with Hamming
distance ≤ 40; and 22 curated groups for crops and different print versions, found by reviewing contact
sheets (CLIP similarity only as an aid). We keep the highest-resolution copy of each artwork.

**Q: What was the upscaling problem?**
26 local files were exactly 384 px on the short side because they had been enlarged from originals as small as
200 px. We fetched the original sizes from Wikimedia and validate against those, so they are excluded.

**Q: Why aspect-ratio buckets instead of square crops?**
Most Varma canvases are tall (~1:1.4). A square centre crop cuts off heads and feet, which teaches bad
anatomy and composition. Buckets such as 448×640 keep ~512² pixels (what SD1.5 expects) with almost no crop.

**Q: Why hold out 12 images?**
To evaluate style similarity against paintings the model has never seen; comparing with training images would
reward memorisation.

**Q: How are captions generated? How reliable are they?**
BLIP-large writes a sentence and BLIP-VQA answers 12 fixed questions (clothing, jewellery, pose, setting…),
with gated follow-ups and filters for meaningless or anachronistic answers. They are not perfect ("red dress" for
a sari), so every answer is saved and captions can be corrected and rebuilt.

**Q: Why does the caption length matter?**
CLIP reads at most 77 tokens; longer text is silently truncated. We assemble fields by priority and stop at
70 tokens so the important parts are always seen.

---

## 6. Evaluation and results [B9, D]

**Q: How do you know the LoRA works?**
Same prompt and same seed with and without the LoRA: the LoRA images move to Varma-like staging (full figures,
pillars, carpets, darker painted backgrounds), frames disappear (5/15 → 0/15), and the validation images show
the style emerging over training.

**Q: What metrics did you use?**
CLIP ViT-B/32 cosine similarity for prompt adherence, style similarity to the 12 held-out paintings, diversity
across seeds, and nearest-training-image similarity as a memorisation check — all compared pairwise with the base
model on identical prompts/seeds.

**Q: The CLIP improvement is tiny (0.669 → 0.681). Isn't that a failure?**
CLIP measures mostly *content* similarity; it barely sees painting technique, framing or anatomy. The visual
changes are clear, so the metric is the weak link, not the model. That is why we also prepared a human rating
sheet and plan a stronger evaluation in Milestone 4.

**Q: Why not FID?**
FID compares feature statistics of large image sets and needs thousands of images to be stable; with 15
generated and 12 reference images it would be meaningless.

**Q: Why did prompt adherence drop slightly (0.328 → 0.317)?**
The difference is within noise for 15 pairs, and a style adapter naturally pulls images toward its training
distribution (e.g. royal settings), which can reduce literal prompt matching a little.

**Q: What problems does the model have?**
A dark moustache-like shadow on some women's upper lips, small signature-like marks in corners (learned from
signatures printed on oleographs), and crowded mythological scenes that still look like posters. Hands remain weak,
as in base SD1.5.

**Q: How would you fix those?**
Name or crop signatures in the training captions (as we did for frames); find which training portraits cause the
lip shadow and re-caption or remove them; compare checkpoints and LoRA scales with human ratings; ControlNet in
Milestone 4 will help anatomy and poses.

---

## 7. Engineering and challenges

**Q: What went wrong during the project?**
(1) The original dataset had placeholder captions and upscaled images; (2) the `runwayml` SD1.5 repository was
removed from Hugging Face — we switched to the official mirror; (3) on Colab, xformers forced a torch downgrade and
bitsandbytes crashed with the new Triton — both removed; Colab's newest image (Python 3.13) also lacked wheels, so
we used runtime 2026.07; (4) on the Mac, fp16 with attention slicing produced black images — fixed by disabling
slicing on MPS.

**Q: Did you find any bugs in your own code?**
Yes — the learning-rate warm-up was multiplied by the accumulation factor (400 instead of 100 steps), the loss was
logged from only one of the four accumulated images, and only the final weights could be loaded. All fixed and
covered by tests.

**Q: How do you test an ML project like this?**
101 automated tests: dataset validation and curation logic, caption rules, bucket sizes, generation settings,
evaluation outputs, and an end-to-end training run on a 1 MB test model that checks freezing, checkpointing,
resume and that the exported LoRA loads.

**Q: Why train on Colab and not your laptop?**
The laptop has 8 GB of shared memory and no CUDA GPU — a benchmark took ~3 s per image, so the 6000-image run would
take 5–6 hours with heavy swapping. A free T4 did it in 58 minutes. Inference does run on the laptop (~1 min per image).

---

## 8. Ethics

**Q: Are these "real" Ravi Varma paintings?**
No. They are AI-generated, Ravi Varma–*inspired* images and are labelled that way in the README and in every image's
metadata.

**Q: Any ethical concerns?**
Style imitation of a historical artist (public domain, so legal, but should be credited and labelled), possible
misuse to pass off fakes, and biases of the base model and the VQA captioner (e.g. Western defaults like "dress").
We record provenance, label outputs and do not claim authenticity.

---

## 9. Future work

**Q: What is Milestone 3?**
An LLM turns a story description ("Yudhishthira weeping over Karna at Kurukshetra") into a detailed, attribute-rich
prompt with the trigger word (characters, clothing, setting, lighting) that fits in 77 tokens.

**Q: What is ControlNet and why add it?**
ControlNet is an extra network that conditions Stable Diffusion on a structural input such as an OpenPose skeleton or
Canny edges. It lets the user fix the pose and layout — addressing weak anatomy and giving Varma's balanced,
theatrical staging.

**Q: What would you do with more time or compute?**
Larger and cleaner dataset (high-resolution museum scans), SDXL at 1024 px, human evaluation with several raters,
and comparing rank / step / LoRA-scale settings systematically.
