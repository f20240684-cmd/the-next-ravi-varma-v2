# 9. Setup, operation and troubleshooting

## 9.1 Requirements

| Item | Needed for | Notes |
|---|---|---|
| Python 3.11 | everything | Newer versions lack wheels for some pins (Colab 3.13 was the reason for pinning its runtime) |
| ~10 GB disk | models cache + data | SD1.5 ≈ 4 GB, ControlNets ≈ 1.4 GB each, BLIP ≈ 2 GB, VAE ≈ 0.3 GB |
| Internet on first run | model downloads from the HF Hub | Afterwards everything runs offline, except the optional LLM |
| Apple-silicon Mac (8 GB+) or a CUDA GPU | generation | CPU works but takes many minutes per image |
| CUDA GPU ≥ 16 GB (Colab T4) | training, hires fix | Not on the Mac |
| No accounts or keys | default use | Optional: an LLM key in `.env` |

## 9.2 Install

```bash
git clone https://github.com/f20240684-cmd/the-next-ravi-varma-v2.git
cd the-next-ravi-varma-v2
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # optional; leave values empty for offline use
```

**Verify the installation:**

```bash
pytest -q                   # expect: all pass, ≤ 2 skipped (CLIP weights not cached)
python scripts/generate.py --prompt "Saraswati playing the veena" --expand --dry-run
```

## 9.3 Everyday commands

| Goal | Command |
|---|---|
| One painting | `python scripts/generate.py --prompt "a royal woman reading a letter" --seed 42 --width 448 --height 640 --face-fix` |
| Base model (no LoRA) | add `--no-lora` |
| Story → prompt only | `--expand --dry-run` |
| Pose transfer | `--pose-image data/reference/0225.jpg --controlnet-type openpose` |
| Photo → painting | `--init-image photo.jpg --strength 0.55 --face-fix` |
| Older checkpoint | `--lora-path checkpoints/lora/ravi_varma_v3/checkpoint-1000` |
| Web app | `python app/app.py` → http://127.0.0.1:7860 |
| Evaluation suite | `make samples && make baseline && make evaluate` |
| Tests / lint | `make test`, `make lint` |
| Retrain | §7.4 Flow 2; `TAG=v4 bash scripts/colab_train.sh` on Colab |

## 9.4 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PROMPT_EXPANSION_PROVIDER` | `none` | `openai` or `anthropic` enables LLM mode |
| `PROMPT_EXPANSION_MODEL` | empty | Model name at that provider (e.g. a Gemini model through `OPENAI_BASE_URL`) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | empty | `<your-key-here>`; never commit it |
| `OPENAI_BASE_URL` | unset | Any OpenAI-compatible endpoint. **Leave it commented out** for real OpenAI, because an empty value breaks the client |
| `HF_TOKEN` | empty | Only for gated models (none are used by default) |
| `MODEL_CACHE_DIR` | `~/.cache/huggingface` | Where weights are cached |

## 9.5 Resetting state

| To reset | Do |
|---|---|
| Generated images | Delete files under `outputs/generated/` (keep the committed subfolders used by the docs) |
| Prepared data | `python scripts/prepare_dataset.py` (it clears `data/processed/` itself) |
| Captions | `python scripts/generate_captions.py --overwrite` (slow) or `--reflatten` (fast, no models) |
| Model cache | Delete `~/.cache/huggingface/hub/models--…` for that model |
| Training run | Use a new `TAG`; old runs stay untouched |

## 9.6 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TypeError: unhashable type: 'dict'` when starting the app | starlette 1.x with Gradio 4.44 | `pip install fastapi==0.115.0` (pinned in requirements) |
| All-black image / `RuntimeError: … all-black image` | fp16 NaN; attention slicing forced on MPS | Keep `attention_slicing: auto`; or `dtype: float32` |
| Mac extremely slow (20+ min per image), heavy swap | Not enough free unified memory (SD + ControlNet ≈ 3.5 GB) | Close other apps; avoid hires fix; use Colab for batches |
| `CUDA out of memory` with `--hires-scale` | Hires needs > 8 GB | Drop `--hires-scale`, or use a 16 GB GPU |
| `LoraCheckpointMissingError` | `lora_path` points to a missing folder | Check `configs/generation.yaml` → `model.lora_path`, or pass `--lora-path` |
| `lora_loaded: false` in the sidecar | `--no-lora`, or a missing path with fallback allowed | See above |
| Style looks weak | No trigger in a custom pipeline, or `lora_scale` low | The generator adds `<rvvarma>` automatically; set the style strength to 1.0–1.2 |
| Prompt details ignored | Prompt over 77 tokens | Use `--expand` (fits 75), or shorten |
| Moustache on women | Negative prompt truncated (old configs) | Keep facial-hair terms first, under 77 tokens |
| Photo mode changes the subject | Strength too high | 0.5–0.6; the Canny lock keeps outlines |
| Faces not sharpened | Profile view or no eye found | Expected Haar limit; use frontal compositions |
| `--expand` ignores the LLM | Provider `none`, missing key, quota error | Check `.env`; the log says "falling back to local" |
| OpenAI client error with Gemini | `OPENAI_BASE_URL` empty or wrong model name | Set the full base URL and a model that the endpoint lists |
| Download 429 errors | Wikimedia rate limit | The script retries; just rerun (it skips existing files) |
| Tests skipped | CLIP weights or the tiny SD model not cached and offline | Run once with internet; `pytest -rs` shows the reasons |
| Colab: `triton.ops` import error | bitsandbytes installed | `pip uninstall bitsandbytes`; use `requirements-colab.txt` |
| Colab: torch downgraded | xformers pin | Don't install xformers; SDPA is used |

## 9.7 Where to look when something fails

1. **Console logs.** `setup_logging` prints stage summaries; secrets are redacted.
2. **The sidecar `.json`** of any image: exact prompt, seed, LoRA file, device, dtype.
3. **`data/metadata/validation_report.json`:** why an image was excluded.
4. **`training_summary.json` and `loss_history.csv`:** what a run actually did.
5. **`pytest -rs`:** which tests were skipped and why.
