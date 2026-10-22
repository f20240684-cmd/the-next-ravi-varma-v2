# 13. Glossary

| Term | Meaning in this project |
|---|---|
| Accelerate | Hugging Face library handling mixed precision, gradient accumulation and checkpoint state in training |
| α (alpha) | LoRA scaling numerator; the update is multiplied by α/r (16/16 = 1 here) |
| Aspect bucket | One of a fixed set of training sizes (e.g. 448×640); each image goes to the nearest one |
| BLIP / BLIP-VQA | Salesforce vision-language models: one writes a caption, the other answers questions about an image |
| Canny | Edge detector; its edge map conditions the Canny ControlNet |
| Checkpoint | Saved training state every 250 steps, including a loadable LoRA file |
| CFG / guidance scale | Classifier-free guidance; how strongly the prompt (vs the negative prompt) steers each step (7.5) |
| CLIP | Contrastive image-text model. ViT-L/14 is SD1.5's text encoder; ViT-B/32 is the evaluation judge |
| Colab T4 | Google Colab's free 16 GB NVIDIA GPU, used for training |
| ControlNet | A network that adds spatial conditioning (pose, edges) to a frozen diffusion UNet |
| Cosine similarity | a·b / (‖a‖‖b‖); 1 = same direction |
| dHash | Difference hash; a perceptual fingerprint for near-duplicate detection |
| Diffusers | Hugging Face library with SD pipelines, schedulers and model classes |
| Diversity | Mean CLIP distance between images of one prompt across seeds |
| DPM-Solver | Fast diffusion sampler (scheduler) used for 30-step generation |
| Face fix | Second-pass redraw of each detected face at 512 px (ADetailer-style) |
| fp16 / fp32 | 16- / 32-bit floating point. Frozen weights fp16, LoRA fp32 |
| Gradio | Python web-UI library used for the app |
| Hamming distance | The number of differing bits between two hashes |
| Hires fix | Upscale then img2img redraw at low strength |
| Hold-out / reference set | The 12 paintings never trained on, used for style scoring |
| img2img | Diffusion starting from a noised version of an existing image |
| Latent | The 8×-downsampled 4-channel representation that diffusion operates on |
| Lexicon | `configs/epic_lexicon.yaml`: characters, aliases, iconography, places |
| LoRA | Low-Rank Adaptation: trainable low-rank matrices added to frozen weights |
| Manifest | `data/processed/manifest.jsonl`: one training row per image (path + caption) |
| Min-SNR-γ | Loss weighting that caps the influence of easy low-noise timesteps |
| MPS | Apple's Metal Performance Shaders backend for PyTorch on Apple silicon |
| Negative prompt | Text the sampler steers away from; also limited to 77 tokens |
| Noise offset | Small constant added to training noise to allow very dark or bright images |
| OpenPose | Body-keypoint detector (18 joints); its skeleton conditions the OpenPose ControlNet |
| Paired evaluation | Comparing LoRA vs base on identical prompts and seeds |
| PCK@0.1 | Percentage of Correct Keypoints within 0.1 × image size |
| PEFT | Hugging Face Parameter-Efficient Fine-Tuning library (provides LoRA) |
| Rank (r) | Inner dimension of the LoRA matrices (16) |
| safetensors | A safe tensor file format without executable pickle code |
| Scheduler | The algorithm that sets denoising step sizes |
| SD1.5 | Stable Diffusion 1.5, the base text-to-image model |
| Sidecar | A `.json` file next to an image or caption recording how it was made |
| Strength | img2img: how much noise is added (0 = unchanged, 1 = new image) |
| Token | A sub-word unit of CLIP's tokenizer; `<rvvarma>` is 5 tokens |
| Trigger token | `<rvvarma>`; the word in every caption that carries the learned style |
| UNet | The noise-prediction network inside SD |
| VAE | Variational autoencoder mapping pixels ↔ latents; `sd-vae-ft-mse` is the decoder used |
| VQA | Visual question answering |
