<!-- Extracted from the technical handbook (docs/handbook/src/); edit the source files and rerun build_handbook.py. Commit e2cae62. -->

# 12. Demonstration and interview preparation

> Use only the numbers in this handbook. Each comes from a file in `outputs/evaluations/` or
> `checkpoints/lora/ravi_varma_v3/`. Don't claim users, deployment or results that don't exist.

## 12.1 The 30-second explanation

"We taught Stable Diffusion 1.5 to paint like Raja Ravi Varma using LoRA, a 12.8 MB add-on trained on 182 curated
public-domain paintings on a free Colab GPU. On top of that we built prompt expansion for Indian-epic stories, pose
control with ControlNet, face sharpening and a photo-to-painting mode, all running on an 8 GB Mac. We measured it
against the base model on held-out paintings: style similarity rose from 0.669 to 0.701 with no loss in prompt
adherence, and pose adherence went from 0.17 to 0.86 with ControlNet."

## 12.2 The 2-minute explanation

1. **Problem.** Base SD1.5 gives glossy "calendar art" for Varma-style prompts and even paints gilt frames (5 of 15 images).
2. **Data.** 300 candidate images from Wikimedia Commons and the Wellcome Collection, with licences recorded.
   - We checked resolution against the original scan (26 upscaled copies caught).
   - We removed duplicates: exact, perceptual and curated (7 copies of one print).
   - We excluded non-paintings.
   - Result: 182 training images, 12 held out.
3. **Captions.** BLIP + 12 VQA questions describe the content, so the trigger `<rvvarma>` learns only the style. Frames and printed text are *named* in captions so they don't leak into the style. Everything fits CLIP's 77 tokens.
4. **Training.** LoRA rank 16 on the UNet attention layers: 3.2 M of 860 M parameters, 1500 steps, 58 minutes on a T4. v3 added Min-SNR weighting, noise offset and caption dropout.
5. **Features.** Everything shares one loaded model (that's how it fits in 8 GB):
   - text → painting
   - lexicon-based prompt expansion
   - OpenPose / Canny ControlNet
   - face fix
   - photo → painting with a Canny lock
6. **Evaluation.** Paired against the base model, with the same prompts and seeds, on held-out references:
   - style 0.669 → 0.701 (12 of 15 pairs)
   - no memorisation (max 0.898 < 0.95)
   - PCK 0.17 → 0.86
   - prompt expansion showed **no** CLIP gain, and we report that.

## 12.3 The 5-minute explanation

Use the 2-minute version and add:
- **Architecture (§3.2 diagram).** The weights file is the only hand-off between training and inference. CLI and app share one `RaviVarmaGenerator`.
- **The two bugs that mattered most:**
  - A 100-token negative prompt was silently cut at 77, so "frame, blurry" were never applied. Shortening it fixed a moustache artefact on women in 3 of 5 cases.
  - 24 captions overflowed and lost their artefact notes.
- **The photo-mode failure and fix.** Plain img2img turned a Spider-Man photo into a turbaned man. A Canny ControlNet lock fixed it.
- **Honest limitations** (§12.6) and the next steps: a human study and ablations.

## 12.4 Live demo script (about 6 minutes)

| Step | Action | Say |
|---|---|---|
| 0 (before) | `python app/app.py`; warm up with one image; close other apps | — |
| 1 | **Generate** tab: chip "Damayanti and the swan" (or type "Saraswati playing the veena by a lotus lake"), seed 42, Portrait 448×640, face fix on | "The scene is expanded with standard iconography, within 75 tokens." Show the prompt card |
| 2 | While it renders (~2 min on the Mac) | Explain the trigger token and the LoRA |
| 3 | Show the result and the details card | "Seed, LoRA file and decoder are recorded, so this exact image can be regenerated." |
| 4 | Same scene with style strength 0 | Base-model look, side by side |
| 5 | **Checkpoint Hub** | Loss is flat, which is why checkpoints are chosen by eye |
| 6 | **Photo → Varma** with a prepared portrait | The outline lock; moustache preserved |
| 7 | README results table | The numbers and their caveats |

**Fallback if the Mac is too slow:** show the pre-rendered images in `outputs/evaluations/` and the screenshots in `docs/screenshots/`.

## 12.5 Questions and answers

**Data**

1. *Why so few images?* That's all the public-domain, decent-resolution Varma reproductions with clear provenance. LoRA is designed for small data, and memorisation was checked (max similarity 0.898).
2. *How do you know the images are legal to use?* Every record stores its source file and licence (public domain / CC BY). Varma died in 1906.
3. *How were duplicates found?* In three ways: an exact SHA-256 hash, a 256-bit dHash with Hamming ≤ 40, and curated groups that CLIP suggested and a human confirmed. The highest-resolution copy is kept.
4. *Why check the original resolution?* 26 files had been upscaled to 384 px from originals as small as 200 px. They would pass a local size check but teach blur.
5. *What is the data split?* 182 train, 12 held out, no validation split. Too little data to spare; the checkpoints are judged visually.
6. *Why keep prints?* Genuine compositions, and the data is scarce. They're tagged, so a run without them is a one-line change. Their effect is untested.

**Captions**

7. *Why not use BLIP captions directly?* One vague sentence. The VQA template gives consistent attributes, so content words absorb content and the trigger absorbs style.
8. *What's the 77-token limit?* CLIP's text encoder reads 77 tokens; the rest is silently dropped. Captions are capped at 69 + 6 for the trigger.
9. *How did the frames disappear?* By naming "frame" in captions where one was present, so the trigger didn't absorb it. Measured: 5/15 → 0/15.

**Model and training**

10. *Why SD1.5 not SDXL?* Training on a T4 and inference on an 8 GB Mac; SDXL needs ~3× the memory. SD1.5 also has mature ControlNets.
11. *What does LoRA do mathematically?* W' = W + (α/r)·B·A with rank 16. Only A and B train: 3.2 M parameters, a 12.8 MB file.
12. *Why rank 16?* A common middle ground. It wasn't ablated; that's future work.
13. *Which layers?* The attention projections q, k, v, out, where text meets image.
14. *What's the trigger token for?* It switches the style on or off and collects everything the captions don't describe.
15. *What are Min-SNR, noise offset and caption dropout?* Respectively: they reweight easy timesteps, allow darker or brighter images, and force the trigger alone to carry the style. They were added in v3.
16. *Why does the loss stay flat?* It depends on a random noise level per step; style quality isn't visible in it.
17. *How long did training take?* 3,493 s ≈ 58 min for 1500 steps on a Tesla T4 (`training_summary.json`).
18. *How do you know it's not copying?* The highest CLIP similarity of any output to any training image is 0.898, below the 0.95 threshold.

**Generation**

19. *How does ControlNet work with your LoRA?* ControlNet adds features from a pose or edge map to the frozen UNet. The LoRA changes the attention weights. They're independent, so they combine.
20. *Why OpenPose over Canny?* OpenPose transfers only the pose. Canny traces the whole image (its higher style score comes from copying).
21. *How does it run on 8 GB?* SD1.5 is loaded once, and every pipeline reuses its components; plus fp16, no attention slicing on MPS, and `empty_cache`.
22. *What is the face fix?* Detect faces (Haar + eye check), redraw each at 512 px with img2img strength 0.4, and blend back with a feathered mask.
23. *Why did photo mode change Spider-Man into another person?* Plain img2img at 0.7 kept nothing structural. Locking the Canny edges fixed it.
24. *Why do seeds reproduce across machines?* The noise comes from a CPU generator, which is identical on CUDA, MPS and CPU.

**Evaluation**

25. *Why CLIP and not FID?* FID needs thousands of images; we have 15 pairs. CLIP similarity to held-out paintings is usable at small n, but coarse.
26. *Why is the evaluation paired?* Absolute CLIP scores mix content and style. Same prompt + seed isolates the LoRA's effect.
27. *Is 0.669 → 0.701 significant?* It's consistent (12 of 15 pairs), but no significance test was run, and n is small. The images are the stronger evidence.
28. *What is PCK?* The share of reference joints reproduced within 10 % of the image size. LoRA only 0.17, with OpenPose 0.86, on 6 references.
29. *Did prompt expansion help?* Not measurably: CLIP 0.326 raw vs 0.317 local vs 0.320 LLM on 4 scenes. It helps single iconic figures visually; complex actions fail either way.
30. *What got worse in v3?* Seed diversity, 0.119 → 0.092.

**Engineering**

31. *How is it tested?* 121 tests, including a real LoRA training run on a 1 MB tiny SD model. CI on every push.
32. *What isn't tested?* Real GPU training, real image quality, real ControlNet / BLIP loading, the UI layout.
33. *How are secrets handled?* `.env` is git-ignored with an empty example; keys are only read from env; logs redact key-like lines. No key is needed by default.
34. *What would you do with more time?* A human study, ablations, a Hugging Face Space, an SDXL port, a better face detector.
35. *What was your contribution?* (Answer from your own role in the team. This handbook doesn't attribute individual work.)

## 12.6 Weaknesses to state before you are asked

- **Hands and crowds** remain poor (an SD1.5 limit). Pose control fixes the count, not the hands.
- **Complex actions** fail with or without expansion.
- **Small samples:** 15 pairs, 6 pose references, 4 expansion scenes. No human study.
- **Confounded v3:** data and training changes went into one run.
- **Native resolution ~448×640.** Hires needs a bigger GPU.
- **Haar face detection** misses profiles.
- **Diversity dropped** in v3.
- **Stale documents** remain in the repo (Appendix A).
