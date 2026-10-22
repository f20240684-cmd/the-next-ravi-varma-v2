# 14. Quick reference

| | |
|---|---|
| **What** | SD1.5 + LoRA (r16) style model of Raja Ravi Varma, with prompt expansion, ControlNet, face fix, photo mode, evaluation, a Gradio app |
| **Data** | 300 candidates → 194 accepted → 182 train + 12 held out; Commons + Wellcome; provenance in `metadata.jsonl` |
| **Training** | Colab T4, 1500 steps, 58 min, fp16 frozen + fp32 LoRA, grad-accum 4, Min-SNR 5, noise offset 0.05, caption dropout 0.1 |
| **Weights** | `checkpoints/lora/ravi_varma_v3/pytorch_lora_weights.safetensors` (12.8 MB, in git) |
| **Results** | Style 0.669 → 0.701 (12/15); top-3 0.734 → 0.770; prompt 0.328 → 0.330; diversity 0.106 → 0.092; max training sim 0.898; PCK 0.17 → 0.86 |
| **Null result** | Prompt expansion: CLIP 0.326 / 0.317 / 0.320 (raw / local / LLM) |
| **Tests** | 121 (119 + 2 skipped on CI) |

**Commands**

```bash
source .venv/bin/activate
python scripts/generate.py --prompt "SCENE" --expand --seed 42 --width 448 --height 640 --face-fix
python scripts/generate.py --prompt "SCENE" --pose-image REF.jpg --controlnet-type openpose
python scripts/generate.py --init-image PHOTO.jpg --strength 0.55 --face-fix
python app/app.py                       # http://127.0.0.1:7860
pytest -q && python -m pyflakes src scripts app tests
```

**Files to know**

| File | Why |
|---|---|
| `src/ravi_varma/generation/pipeline.py` | All generation modes |
| `src/ravi_varma/training/lora_trainer.py` | The training loop |
| `src/ravi_varma/data/validation.py` | What gets into the dataset |
| `configs/generation.yaml` | LoRA path, VAE, negative prompt, defaults |
| `configs/dataset.yaml` | Curation, duplicates, notes, hold-out |
| `docs/DECISIONS.md`, `docs/RESULTS.md` | Reasons and the full results record |

**Key numbers:** 77 CLIP tokens · captions 69 + 6 · prompts ≤ 75 · rank 16 / α 16 · 3.2 M params (0.37 %) ·
448×640 · 30 steps · CFG 7.5 · face fix 512 px at strength 0.4 · photo strength 0.55 · dHash ≤ 40/256 ·
PCK threshold 0.1 · copy threshold 0.95.
