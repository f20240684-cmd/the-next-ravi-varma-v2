# Design decisions: The Next Ravi Varma

This file records why the project is built the way it is. Each entry gives:
- **the decision**
- **the alternatives we considered**
- **why we chose it**
- **the evidence** (a number, a test, or what we saw) wherever one exists

Use it to answer "why did you do X and not Y?" in a viva, or before changing something.

How to read it:
- "Measured" means we ran it and the result is in `outputs/`.
- "Expected" means standard practice that we did not verify separately on this project.

---

## 1. Overall approach

### 1.1 Stable Diffusion 1.5 as the base model (not SDXL)
- **Alternatives:** SDXL (1024 px, better faces and hands), a newer DiT model, training from scratch.
- **Why:**
  - **Compute:** SD1.5 trains on a free Colab T4 (16 GB) in about an hour and generates on the team's 8 GB MacBook. SDXL needs about 3× the memory and would not run locally at all.
  - **Ecosystem:** pre-trained SD1.5 ControlNets (OpenPose, Canny) exist, which Milestone 4 needs.
  - **The proposal** names SD1.5 or SDXL, so both are in scope.
  - **Training from scratch** needs millions of images; we have under 200.
- **Cost of the choice:** native resolution is about 512 px, so faces in full-body shots are only about 40 px and come out soft. We addressed that in section 7 instead of switching models.
- **Revisit if:** a GPU with 24 GB or more is available. SDXL is the single biggest quality step left.

### 1.2 LoRA instead of full fine-tuning or DreamBooth
- **Alternatives:** full UNet fine-tuning (860 M parameters), DreamBooth, textual inversion.
- **Why:**
  - **Overfitting:** with 122–182 paintings, updating all 860 M weights would memorise them.
  - **Size:** LoRA trains 3.2 M parameters (0.37 %) and the result is a 12.8 MB file instead of 4 GB.
  - **Flexibility:** it can be switched on and off and scaled at generation time (`--lora-scale`).
  - **Textual inversion** learns only a word embedding. It is too weak to change rendering technique (brushwork, skin, palette).
- **Evidence (measured):** no memorisation. The highest similarity of any generated image to any training image is 0.884 (v2) and 0.898 (v3), below the 0.95 copy threshold.

### 1.3 Rank 16, alpha 16, attention projections only
- **Alternatives:** rank 4/8/32/64; also adapting feed-forward or convolution layers.
- **Why:**
  - Rank 16 is the common middle ground for a style LoRA on a small dataset.
  - alpha = rank gives scale α/r = 1, so the update is neither damped nor amplified.
  - The attention layers (`to_q/k/v/out`) are where the text conditions the image, so they are the natural place to tie a style to a trigger word.
- **Not done:** a rank ablation (8/16/32). That is listed as future work; the choice is *expected* to be reasonable, not measured.

### 1.4 A trigger token, `<rvvarma>`
- **Why:** every training caption starts with `<rvvarma>, ...` and describes the *content* (people, clothes, pose). What the captions don't explain (oil texture, palette, lighting, staging) ends up attached to the one word present in every caption. At generation time the style is on with the trigger and off without it, which also gives a clean base-vs-LoRA comparison.
- **Detail:** CLIP splits `<rvvarma>` into 5 tokens. That doesn't matter; the LoRA learns that combination.

---

## 2. Dataset

### 2.1 Sources: Wikimedia Commons + Wellcome Collection only
- **Alternatives considered and rejected:**
  - **Google Arts & Culture:** has high-resolution images, but its terms forbid downloading.
  - **Museum and gallery websites:** no open licence or API.
  - **The Met, Cleveland Museum of Art, Art Institute of Chicago:** we checked their open-access APIs. Cleveland and AIC have no Ravi Varma works. The Met has 4 public-domain items, all prints, mostly not attributed to him.
- **Why:**
  - Both sources have an open API and a per-image licence, so every image has recorded provenance (`metadata.jsonl`).
  - Ravi Varma died in 1906, so his works are public domain.
  - The Wellcome Collection adds about 50 chromolithographs, almost all marked Public Domain.
- **Reproducibility:** `download_dataset.py --from-metadata` rebuilds the identical raw set. Images are not committed to git; their exact sources are.

### 2.2 Check resolution against the original, not the local file
- **What we found:** 26 local files were exactly 384 px wide because they had been *upscaled* from originals as small as 200 px. Upscaling adds blur, not detail.
- **Decision:** validation checks the original Commons size recorded at download (`source_width/height`), so an upscaled copy cannot hide a low-resolution original.
- **Second download:** `--min-side 800` skipped small originals before downloading them. The new images have a median short side of 2,400 px, against 605 px for the first 200.

### 2.3 Download 2048-px renditions, not originals
- **Why:** some Commons originals are 100 MB+ TIFF scans. Training uses 448–640 px, so a 2048-px rendition loses nothing and keeps downloads fast. The *original* size is still what gets recorded and validated.

### 2.4 Aggressive duplicate removal (keep the sharpest copy)
- **Problem:** the same painting exists as an oil scan, several oleograph print runs and crops. Commons had **7 copies of the Saraswati oleograph**.
- **Why it matters:** a painting that appears 7 times is learned 7 times as strongly. The LoRA then over-learns a few compositions and loses variety.
- **How:**
  1. Exact file hash.
  2. Automatic perceptual hash (dHash, Hamming distance ≤ 40).
  3. Curated duplicate groups. For the v3 additions these were found with **CLIP image similarity** (pairs above 0.92), and **every pair was confirmed by eye**. Three similar-looking pairs turned out to be different paintings and were kept.
- In each group the highest-resolution copy is kept. Groups live in `configs/dataset.yaml`, so the decision can be audited.

### 2.5 Exclude non-paintings by hand
- Postage stamps, gallery photos, multi-panel grids, pages with large text, and a photo of an ivory sculpture.
- **Why:** each would teach the model the wrong thing (stamp borders, panel layouts, photographic lighting). There were only 11, so a human list (`curation.exclude`, with a reason each) is simpler and more accurate than a classifier.

### 2.6 Keep the Wellcome prints, but tag them
- **Trade-off:** prints are the source of the flat colours and signature marks we saw in v2. But they are genuine Varma compositions and the dataset is small.
- **Decision:**
  - Include the 29 accepted prints.
  - Tag them `"medium": "chromolithograph"`, so a future run can drop them in one line.
  - Name their printed title text in the caption (2.9).
- **Open question:** whether prints help or hurt the style. That would need a run without them (future work).

### 2.7 Hold out 12 paintings for evaluation
- **Why:** style similarity measured against *training* images would reward memorisation. 12 paintings, chosen to span portraits, groups, deities, night scenes and a male portrait, are never trained on.
- **Why 91/9 rather than 80/20:** with under 200 images, every training image counts, and 12 references are enough for a similarity score. There is no separate validation split. "Validation" during training means rendering 3 fixed prompts at each checkpoint so progress can be judged by eye.
- **Kept unchanged** across v2 and v3, so the two runs are comparable.

### 2.8 Aspect-ratio buckets instead of square crops
- **Why:** most Varma paintings are portrait format. Square centre crops would cut off heads and feet. Images are resized into buckets (448×640, 448×576, 640×448, …; multiples of 64) and batches are drawn per bucket.
- **Evidence:** 122 of the 182 v3 training images fall in the 448×640 bucket, which is why 448×640 is the default generation size.

### 2.9 Name unwanted marks in the caption instead of hiding them
- **Problem:** v1 learned to paint a gilt **frame**, because frames appeared in training images and the trigger word absorbed them.
- **Decision:** name the artefact in that image's caption (`caption_notes`): "inside a red decorative frame", or "printed title text in the bottom margin" for prints. The word "frame" or "text" then absorbs the artefact instead of the trigger, and the negative prompt excludes it at generation time.
- **Evidence (measured):** frames dropped from 5/15 images (base) to **0/15** (v2).
- **Bug found and fixed (v3):** notes used to be appended *after* the caption had been cut to fit CLIP, so long captions lost them. Notes are now placed inside the token budget, right after the title. All 194 captions were verified to fit.

### 2.10 Template captions: BLIP sentence + 12 VQA questions
- **Alternatives:** BLIP sentence only; a large VLM (LLaVA); hand-written captions.
- **Why:**
  - BLIP alone gives one vague sentence.
  - A fixed set of questions (people count, clothing, colour, jewellery, pose, action, expression, objects, setting, time of day) gives consistent, attribute-rich captions. The model then learns to tie *content* words to content, which leaves *style* for the trigger.
  - Both BLIP models run locally on the Mac.
- **Safeguards:**
  - Follow-up questions are gated (no "which jewellery?" if the answer was "no jewellery").
  - Implausible answers ("phone", "tie" in a 19th-century painting) are filtered.
  - Every raw answer is saved, so captions can be re-assembled without re-running the model (`--reflatten`).

### 2.11 Captions must fit CLIP's 77 tokens
- **Why:** CLIP silently ignores everything after 77 tokens, so an over-long caption trains on a truncated sentence.
- **How:** captions are capped at 69 tokens, plus the 6-token trigger prefix and the start/end tokens. Long Commons "titles" (some are whole paragraphs, some contain Malayalam script or Wikidata markup) are cut to their first sentence.
- **Evidence:** 24 of 182 v3 captions overflowed before this fix; 0 do now.

---

## 3. Training

### 3.1 Train on Colab, never on the Mac
- **Why:** an 8 GB Apple-silicon Mac has no CUDA, no fp16 training kernels that matter here, and would take many hours. A free Colab T4 trains 1500 steps in 58 minutes.
- **How:** `scripts/colab_train.sh` runs, in order: environment checks, a 20-step smoke test, full training, sample images, evaluation, and packaging of the results.

### 3.2 Pin the Colab runtime and remove fragile packages
- **Runtime 2026.07:** the newest Colab image is Python 3.13, and our pinned libraries had no packages for it.
- **bitsandbytes removed:** it crashed with Colab's Triton (`triton.ops`). 8-bit Adam would only save about 19 MB for a 3.2 M-parameter LoRA.
- **xformers removed:** its pin forced a torch downgrade.

### 3.3 Smoke test before every long run
- **Why:** 20 steps, written to throw-away folders, take about a minute. On the first Colab run this caught the bitsandbytes crash in 1 minute instead of after an hour.

### 3.4 fp16 frozen model, fp32 LoRA weights; warm-up; checkpoints every 250 steps
- **Mixed precision:** the frozen 860 M UNet in fp16 halves memory; the small trainable LoRA stays in fp32 so updates don't underflow.
- **Warm-up (100 steps):** avoids early instability. We found and fixed a bug where warm-up was counted in micro-batches instead of optimizer steps.
- **Checkpoints:** saved every 250 steps, each also as a loadable LoRA file. That enables resume after a Colab disconnect and lets us pick a checkpoint by eye (the Checkpoint Hub).

### 3.5 v3 additions: Min-SNR weighting, noise offset, caption dropout
- **Min-SNR-γ (γ = 5):** down-weights low-noise timesteps that otherwise dominate the loss. The published result is faster convergence and cleaner detail.
- **Noise offset (0.05):** lets the model learn very dark or very bright images, which suits Varma's dark backgrounds.
- **Caption dropout (10 %):** one step in ten sees only `<rvvarma>`, which pushes the style into the trigger word.
- **All three are off by default in code**, so older configs behave exactly as before. They are switched on in `lora_sd15.yaml`, and tests run them on a tiny model.
- **Kept the same:** rank, learning rate, steps and batch size. That way only the data and these three changes differ between v2 and v3. We did not want to change hyperparameters blindly.
- **Limitation:** the new data and the training changes went into the same run, so their effects can't be separated without an ablation run.

### 3.6 Separate output folders per run (`TAG=v3`)
- **Why:** the results zip is unpacked over the project. Without a tag, v3 would have overwritten the v2 LoRA and v2 images, and the comparison would have been impossible.

---

## 4. Generation

### 4.1 Negative prompt kept under 77 tokens, problem words first
- **Bug found:** the configured negative prompt was 100 tokens, so its tail ("frame, blurry, lowres, …") was **never applied**.
- **Fix:** cut it to 66 tokens with the most important terms first (moustache/facial hair, which fixed the v2 lip shadow, then anatomy, then frame/text/blur). A test now fails if it grows past 77.

### 4.2 Apple-silicon settings
- **fp16 without attention slicing:** fp16 combined with attention slicing produced NaN values (all-black images) on MPS. The generator also refuses to save an all-black image.
- **`torch.mps.empty_cache()` between images:** without it, 8 GB of unified memory filled up and the Mac started swapping.

### 4.3 Seeds use a CPU random generator
- **Why:** the same seed gives the same starting noise on CUDA, MPS and CPU. Images from Colab and the Mac are then comparable, and every image's `.json` sidecar is enough to regenerate it.

### 4.4 Fine-tuned VAE (`sd-vae-ft-mse`) as the default decoder
- **Why:** it uses the same latent space as SD1.5's decoder (so the LoRA is unaffected) and is fine-tuned to decode faces and fine detail more cleanly. The change costs nothing.
- **Evidence:** a visual side-by-side showed a small improvement. This was not measured with a metric. `vae: null` restores the old look.

---

## 5. Milestone 3: prompt expansion

### 5.1 A local lexicon first, the LLM optional
- **Alternatives:** an LLM only (as the proposal suggests), or no expansion.
- **Why the lexicon:**
  - **Deterministic, free and offline:** it works in a viva without internet or an API key.
  - **Reliable iconography:** an LLM's description of Krishna changes from run to run; the lexicon always gives "blue-skinned, peacock-feather crown, yellow dhoti" (27 characters, 18 places, editable YAML).
- **Why the LLM is still there:** it adds setting and pose phrases the lexicon can't. It is grounded: lexicon looks are inserted deterministically and passed to the LLM as context; the user's sentence is kept verbatim; the configured negative prompt cannot be replaced. Any failure falls back to local mode.
- **Any OpenAI-compatible API:** the team had no paid credits, so Google Gemini's free tier works through `OPENAI_BASE_URL`.

### 5.2 Priority-ordered assembly inside 75 tokens
- **Why:** an earlier version produced about 120-token prompts, so most of the expansion was silently ignored.
- **Order:** trigger → the user's scene (verbatim) → character looks → setting → clothing → pose → lighting → composition → style. Parts are dropped from the end until the prompt fits.

### 5.3 Report that it did not measurably help
- **Measured:** CLIP similarity to the story was 0.326 raw, 0.317 local, 0.320 LLM on 4 scenes, which is within noise.
- It clearly helps single iconic figures, but complex *actions* fail either way.
- **Why report it:** the honest result is more defensible than a claimed gain. It is also the reason Milestone 4 adds pose control.

---

## 6. Milestone 4: ControlNet, evaluation, web panel

### 6.1 OpenPose as the main control (Canny as an option)
- **OpenPose** keeps only the body pose (18 joints per person). Clothing, faces and background stay free, so it transfers a *pose* without copying the reference.
- **Canny** keeps every outline, so it nearly traces the reference. It is useful, but it is not "pose transfer".
- We used pre-trained SD1.5 ControlNets; we did not train them.

### 6.2 Share the loaded model instead of loading SD1.5 twice
- **Problem:** the original scaffold built the ControlNet pipeline by loading a second full copy of SD1.5, which an 8 GB Mac can't hold.
- **Decision:** build the ControlNet, img2img and face-fix pipelines from the *already loaded* UNet (LoRA included), VAE and text encoder.
- **Evidence:** ControlNet runs on the Mac at about 1.5–2 min per image.
- **Detail:** the reference image is cropped to the output's aspect ratio first, so poses aren't stretched.

### 6.3 A pose-adherence metric (PCK)
- **Why:** CLIP can't tell whether a pose was followed. PCK, a standard pose-estimation metric, runs OpenPose on the reference and the output and counts the share of reference joints within 0.1 of the image size.
- **Fair setup:**
  - Pose references are held-out paintings.
  - The style score excludes the painting used as the reference, otherwise Canny, which traces it, would score itself.
- **Measured:** PCK **0.17 → 0.86** with OpenPose, with almost no style cost (0.704 → 0.692).
- **Limitations:** OpenPose judges OpenPose-guided images; the metric ignores hands; 6 images × 1 seed.

### 6.4 Gradio for the web panel, styled from a Figma design
- **Why Gradio:** it is Python-native, so it calls the same generator code, and the proposal names it.
- **Why the Figma step:** the default look was plain. We designed the screens in Figma (dark "gallery at night" theme, gilt frame for the result, checkpoint hub) and rebuilt them with a theme and CSS.
- **Pinned `fastapi==0.115.0`:** Gradio 4.44 crashes with newer starlette. Found when the app wouldn't load.
- **App defaults:** 448×640 (the main training bucket); seed 42 (the original "blank = random" field silently sent 0); short example chips that fill in full scenes.

---

## 7. Image quality

### 7.1 Fix faces with a second pass (like ADetailer), not a bigger model
- **Diagnosis:** at 448×640 a full-body face is about 40 px. SD1.5 can't draw eyes and lips that small, so faces look mushy.
- **Decision:**
  1. Find each face with OpenCV's face detector (already installed).
  2. Crop it with some margin.
  3. Redraw it at 512 px with img2img (strength 0.4, "detailed face" + the same prompt).
  4. Blend it back with a feathered oval mask.
  It reuses the loaded model, so it runs on the Mac (about +20 s per face).
- **Why OpenCV's Haar detector:** no new dependency. It misses profiles and occasionally fires on a fold of cloth. Redrawing a false hit is harmless, and a stricter setting lost a real face, so we kept it.
- **Evidence:** the before/after crops (`outputs/evaluations/face_fix_compare.jpg`) show clear, symmetric eyes and defined lips.

### 7.2 Hires fix only on GPU
- Upscale 1.5×, then redraw at 0.4 strength. On the 8 GB Mac it ran out of memory at 1.5× and swapped heavily at 1.25×, so it runs in the Colab script and not by default.
- **Evidence:** v3 hires + face-fix images are visibly sharper (`v3_hires_face_crops.jpg`).

---

### 7.3 Photo → Ravi Varma: img2img with a Canny ControlNet lock
- **Alternatives:** classic neural style transfer (copies colours and texture, not Varma's way of painting skin and cloth); plain img2img; training an image-to-image model.
- **First version, plain img2img: failed.** A Spider-Man photo at strength 0.7 came back as a different picture, a turbaned man in a new pose. Two causes:
  1. Nothing locked the structure.
  2. The default prompt ("portrait of a person, traditional Indian attire") fought the photo.
- **Fix:**
  - img2img plus a **Canny ControlNet on the photo's own edges** (diffusers' `StableDiffusionControlNetImg2ImgPipeline`, built from the already-loaded model so it fits the Mac).
  - A **neutral prompt**, with only the user's optional description added.
- **Evidence (T4):** both test photos keep pose, composition and likeness: the Ravi Varma photograph (face, moustache, turban, medal, cane) and Spider-Man (crouch, pillar, buildings).
- **Default strength 0.55:** with the outlines locked, 0.5–0.7 is safe.
- **Two bugs found along the way:**
  1. The facial-hair negative terms erased a real moustache, so photo mode drops them.
  2. The face fix painted a face onto a medal, so detections now need an eye.
- **Known limits:**
  - Masks and modern costumes are reinterpreted, because the LoRA has only seen Varma paintings.
  - On the 8 GB Mac, SD + ControlNet needs other apps closed or it swaps (20 min per image instead of about 1–2).

## 8. Evaluation philosophy

- **Always paired against the base model:** same prompts, same seeds. Absolute CLIP scores mix content with style, so only the difference is meaningful.
- **Held-out references, a memorisation check, and a diversity score** alongside the style score, so a "better" number can't come from copying or collapsing.
- **CLIP is a weak judge of painting style.** We also show image grids, and `qualitative_review.csv` provides a human 1–5 rating sheet. A blind human survey is the recommended next step.
- **No fabricated results:** every number in the README comes from a file in `outputs/evaluations/`. Failures are reported, including Milestone 3 showing no gain, the v3 diversity drop, Haar false positives, and broken anatomy in group scenes.

---

## 9. What we deliberately did not do (yet)

| Not done | Why | When it would make sense |
|---|---|---|
| SDXL | Doesn't fit an 8 GB Mac; needs a bigger GPU | With 24 GB+ |
| Training the text encoder too | Overfitting risk with < 200 images; the trigger already works | If the trigger proves too weak |
| Rank ablation (8/16/32) | Costs ~1 h of Colab per run | For the final report |
| Separating data vs training fixes in v3 | Another run needed | If a reviewer asks which change helped |
| Training without the Wellcome prints | Another run | To test whether prints flatten the style |
| Publishing on Hugging Face / Spaces | Needs the team's account; public by nature | When the team agrees |
| A face-detection network | Haar works for most faces; no new dependency | If profiles and crowds matter |
